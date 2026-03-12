"""
draft_node.py - UAV 端 Draft 节点（小模型推理）

包含:
  - PyTorchDraftNode: PyTorch 后端 (CUDA / MPS / CPU)
  - VLLMDraftNode:    vLLM 后端 (NVIDIA GPU 高性能推理)
  - MLXDraftNode:     MLX 后端 (Apple Silicon 原生加速)
  - detect_framework(): 自动检测推理框架
  - create_draft_node(): 工厂函数
"""

import platform
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from dssd_utils import sample, tensor_nbytes, resolve_device, get_device_info


# ============ 框架自动检测 ============

def detect_framework(device_str: str, framework_override: str = "auto") -> str:
    """
    根据设备和硬件环境自动选择推理框架
    返回: "mlx" 或 "pytorch"
    """
    # 用户强制指定
    if framework_override != "auto":
        if framework_override == "mlx":
            try:
                import mlx.core  # noqa: F401
                import mlx_lm    # noqa: F401
                return "mlx"
            except ImportError:
                print("[Framework] WARNING: --framework mlx 但 mlx-lm 未安装，回退到 pytorch")
                print("[Framework] 安装方法: pip install mlx-lm")
                return "pytorch"
        return "pytorch"

    # 自动检测
    device_str = device_str.strip().lower()

    # CUDA 设备 → 一定用 PyTorch
    # 注意: 用 _has_nvidia_gpu_no_init() 代替 torch.cuda.is_available()
    # 避免提前初始化 CUDA 上下文 (vLLM 多卡 fork 需要未初始化的 CUDA)
    if device_str.startswith("cuda") or (device_str == "auto" and _has_nvidia_gpu_no_init()):
        return "pytorch"

    # Apple Silicon (MPS) → 优先尝试 MLX
    is_apple_silicon = (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
    )
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if device_str in ("auto", "mps") and (is_apple_silicon or mps_available):
        try:
            import mlx.core  # noqa: F401
            import mlx_lm    # noqa: F401
            return "mlx"
        except ImportError:
            return "pytorch"

    return "pytorch"


# ============ PyTorch 后端 ============

class PyTorchDraftNode:
    """PyTorch 后端的 UAV draft 节点（支持 CUDA / MPS / CPU / 多卡）"""

    def __init__(self, model, device: torch.device, args):
        self.device = device
        self.framework = "pytorch"
        # 如果模型已经通过 device_map 分布到多卡, 不再 .to(device)
        if hasattr(model, "hf_device_map"):
            self.model = model
        else:
            self.model = model.to(device)
        self.model.eval()
        self.args = args
        if hasattr(model, "hf_device_map"):
            devices_used = set(model.hf_device_map.values())
            print(f"[UAV] Draft model distributed across: {devices_used} (PyTorch multi-GPU)")
        else:
            print(f"[UAV] Draft model loaded on: {get_device_info(device)} (PyTorch)")

    def draft_step_DSD(self, prefix: torch.Tensor, gamma: int):
        """
        DSD Draft: 生成 gamma 个候选 token + 完整 logits 分布
        返回: x_draft, q_step_logits (gamma, V), raw_bytes
        """
        x = prefix.to(self.device)
        q_stack = []

        with torch.no_grad():
            for _ in range(gamma):
                logits = self.model(x).logits
                q_stack.append(logits[0, -1].cpu())
                next_tok = sample(logits[:, -1, :],
                                  self.args.temperature,
                                  self.args.top_k,
                                  self.args.top_p)
                x = torch.cat((x, next_tok), dim=1)

        q_step_logits = torch.stack(q_stack, dim=0)
        raw_bytes = tensor_nbytes(q_step_logits)
        return x, q_step_logits, raw_bytes

    def draft_step_DSSD(self, prefix: torch.Tensor, gamma: int):
        """
        DSSD Draft: 生成 gamma 个候选 token + 概率值（上行）+ 完整分布（本地保存）
        返回: x_draft, q_values (list[float]), q_probs (gamma, V), dup_bytes
        """
        x = prefix.to(self.device)
        q_probs = []
        q_values = []

        with torch.no_grad():
            for _ in range(gamma):
                logits = self.model(x).logits
                q_dist = F.softmax(logits[0, -1], dim=-1).cpu()
                q_probs.append(q_dist)

                next_tok = sample(logits[:, -1, :],
                                  self.args.temperature,
                                  self.args.top_k,
                                  self.args.top_p)
                x = torch.cat((x, next_tok), dim=1)

                tok_id = next_tok.item()
                q_values.append(q_dist[tok_id].item())

        q_probs = torch.stack(q_probs, dim=0)
        token_bytes = x.numel() * 4
        prob_bytes = len(q_values) * 4
        dup_bytes = token_bytes + prob_bytes
        return x, q_values, q_probs, dup_bytes

    def resample_DSSD(self, j: int, pj: torch.Tensor, q_probs: torch.Tensor) -> torch.Tensor:
        """DSSD Resample: 用本地 Q_j 和 BS 返回的 P_j 重新采样（兼容 MPS）"""
        q_j = q_probs[j - 1]
        diff = (pj - q_j).clamp(min=0)
        diff = diff / diff.sum()
        if diff.device.type == "mps":
            xj_prime = torch.multinomial(diff.cpu(), 1).unsqueeze(0)
        else:
            xj_prime = torch.multinomial(diff, 1).unsqueeze(0)
        return xj_prime


# ============ vLLM 后端 ============

class _VLLMModelProxy:
    """
    为 EnergyMonitor 提供兼容接口的 vLLM 模型代理。

    EnergyMonitor 需要:
      - model.config          → HuggingFace PretrainedConfig (用于 _extract_model_arch)
      - model.parameters()    → 用于 _count_model_params

    vLLM 的 LLM 对象不直接暴露这些，本代理从 model_config.hf_config 获取 config,
    并根据 config 中的参数量和 dtype 估算参数统计。
    """

    def __init__(self, llm):
        """
        Args:
            llm: vllm.LLM 实例
        """
        self._llm = llm
        # hf_config 就是 HuggingFace 的 PretrainedConfig
        # vLLM 0.8.1: llm.model_config.hf_config
        # vLLM 0.8.5+: llm.llm_engine.get_model_config().hf_config
        if hasattr(llm, 'model_config'):
            self.config = llm.model_config.hf_config
        else:
            self.config = llm.llm_engine.get_model_config().hf_config

    def parameters(self):
        """
        vLLM 不暴露 model.parameters()，
        我们创建一个假的参数迭代器，基于 hf_config 估算参数量。
        返回一组 fake tensor，使 _count_model_params 能正确统计。
        """
        # 从 hf_config 估算总参数量
        cfg = self.config
        hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", 0)
        n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", 0)
        inter = getattr(cfg, "intermediate_size", None) or getattr(cfg, "n_inner", None) or (hidden * 4)
        n_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", 0)
        n_kv_heads = getattr(cfg, "num_key_value_heads", None) or n_heads
        head_dim = getattr(cfg, "head_dim", None) or (hidden // n_heads if n_heads else 0)
        vocab_size = getattr(cfg, "vocab_size", 0)

        if hidden == 0 or n_layers == 0:
            return iter([])

        # 估算每层参数量:
        #   QKV: hidden * (n_heads + 2 * n_kv_heads) * head_dim
        #   O:   hidden * hidden
        #   FFN: hidden * inter * 3 (gate_proj + up_proj + down_proj for SwiGLU)
        #        or hidden * inter * 2 (for standard FFN)
        #   LayerNorm: hidden * 2 (per norm, 2 norms per layer)
        qkv_params = hidden * (n_heads + 2 * n_kv_heads) * head_dim
        o_params = n_heads * head_dim * hidden
        # 假设 SwiGLU (3个 linear) 更常见
        ffn_params = hidden * inter * 3
        norm_params = hidden * 4  # 2 norms, each with weight + bias
        per_layer = qkv_params + o_params + ffn_params + norm_params

        total_params = per_layer * n_layers
        # embedding + lm_head
        total_params += vocab_size * hidden * 2  # embed + lm_head (可能 tied)

        # 确定 dtype
        dtype_str = getattr(cfg, "torch_dtype", "float16")
        if isinstance(dtype_str, torch.dtype):
            dt = dtype_str
        else:
            dtype_map = {
                "float16": torch.float16, "bfloat16": torch.bfloat16,
                "float32": torch.float32, "int8": torch.int8,
            }
            dt = dtype_map.get(str(dtype_str), torch.float16)

        # 返回一个 fake tensor，使 _count_model_params 能统计
        fake = torch.empty(total_params, dtype=dt, device="meta")
        return iter([fake])


class VLLMDraftNode:
    """vLLM 后端的 UAV draft 节点 (NVIDIA GPU 高性能推理)"""

    def __init__(self, llm, tokenizer, args):
        """
        Args:
            llm:       vllm.LLM 实例
            tokenizer: HuggingFace tokenizer
            args:      CLI 参数
        """
        self.framework = "vllm"
        self._llm = llm
        self._tokenizer = tokenizer
        self.args = args
        self.device = torch.device("cuda:0")

        # 为 EnergyMonitor 提供兼容的 model 接口
        self.model = _VLLMModelProxy(llm)

        print(f"[UAV] Draft model loaded with vLLM engine")
        # 兼容 vLLM 0.8.1 和 0.8.5+
        if hasattr(llm, 'model_config'):
            _model_name = llm.model_config.model
        else:
            _model_name = llm.llm_engine.get_model_config().model
        print(f"[UAV] Model: {_model_name}")

    def generate_ar(self, input_ids: torch.Tensor, max_tokens: int,
                    temperature: float = 0.7, top_k: int = 10,
                    top_p: float = 0.0) -> tuple:
        """
        vLLM 自回归生成。

        Args:
            input_ids:   输入 token IDs (torch.Tensor, shape [1, seq_len])
            max_tokens:  生成 token 数
            temperature: 采样温度
            top_k:       top-k 采样
            top_p:       top-p (nucleus) 采样

        Returns:
            (output_ids, generated_token_count)
            output_ids: torch.Tensor, shape [1, seq_len + generated]
            generated_token_count: int, 实际生成的 token 数
        """
        from vllm import SamplingParams

        # 将 input_ids tensor 转为 list
        prompt_token_ids = input_ids[0].tolist()

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            top_k=top_k if top_k > 0 else -1,
            top_p=top_p if top_p > 0 else 1.0,
        )

        # vLLM generate 接受 prompt_token_ids
        outputs = self._llm.generate(
            prompts={"prompt_token_ids": prompt_token_ids},
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        # 解析输出
        output = outputs[0]
        generated_ids = list(output.outputs[0].token_ids)
        generated_count = len(generated_ids)

        # 拼接: prefix + generated
        full_ids = prompt_token_ids + generated_ids
        output_ids = torch.tensor([full_ids], dtype=torch.long)

        return output_ids, generated_count

    def generate_ar_stepwise(self, input_ids: torch.Tensor, max_tokens: int,
                             tracker,
                             temperature: float = 0.7, top_k: int = 10,
                             top_p: float = 0.0):
        """
        vLLM 0.8.1 逐 step 生成 + TokenEnergyTracker 精确测量每个 token 能耗。

        vLLM 0.8.1 的 LLMEngine.step() 是 **同步** 的:
          step() = 调度 → 执行模型 forward pass → 处理输出 → 返回结果
        因此可以在每个 step() 前后用 Zeus begin_window/end_window
        精确测量单个 token 的 GPU 能耗。

        Args:
            input_ids:   输入 token IDs (torch.Tensor, shape [1, seq_len])
            max_tokens:  最大生成 token 数
            tracker:     TokenEnergyTracker 实例
            temperature: 采样温度
            top_k:       top-k
            top_p:       top-p

        Returns:
            (output_ids, generated_count)
        """
        from vllm import SamplingParams

        prompt_token_ids = input_ids[0].tolist()

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            top_k=top_k if top_k > 0 else -1,
            top_p=top_p if top_p > 0 else 1.0,
        )

        # 获取 LLMEngine 实例
        engine = self._llm.llm_engine

        # 添加请求到引擎
        request_id = f"tok_energy_{id(self)}_{time.monotonic_ns()}"
        engine.add_request(
            request_id=request_id,
            prompt={"prompt_token_ids": prompt_token_ids},
            params=sampling_params,
        )

        # 逐 step 生成, 每个 step 前后测量能耗
        generated_count = 0
        final_output = None

        while engine.has_unfinished_requests():
            pos = generated_count

            # --- 精确测量: begin_token → step() → end_token ---
            tracker.begin_token(pos)
            step_outputs = engine.step()
            tracker.end_token(pos)

            # 统计本次 step 产出的新 token 数
            for output in step_outputs:
                if hasattr(output, 'outputs') and output.outputs:
                    new_count = len(output.outputs[0].token_ids)
                    generated_count = new_count
                    final_output = output

            # 安全上限
            if generated_count >= max_tokens:
                break
            if final_output and final_output.finished:
                break

        # 构造输出
        if final_output and final_output.outputs:
            generated_ids = list(final_output.outputs[0].token_ids)
        else:
            generated_ids = []
        generated_count = len(generated_ids)

        full_ids = prompt_token_ids + generated_ids
        output_ids = torch.tensor([full_ids], dtype=torch.long)
        return output_ids, generated_count

    def generate_ar_stepwise_batch(
        self,
        prompts_token_ids: list,
        max_tokens: int,
        tracker,
        temperature: float = 0.7,
        top_k: int = 10,
        top_p: float = 0.0,
    ):
        """
        vLLM 批量逐 step 生成 + TokenEnergyTracker 精确测量每个 position 的能耗。

        **关键**: 所有请求在第一次 step() 前全部 add_request, 确保在同一个 batch 中。

        执行流程:
          vLLM 的调度器会将所有请求放入同一 batch:
            - Step 0: 所有请求做 prefill + 产出第 1 个 token (position 0)
              (如果 prompt 总 token 数很大, 可能需要多个 step 完成 prefill)
            - Step 1+: 所有 active 请求做 decode, 各产出 1 个 token
              position 完全对齐

          每个 step 的能耗 / active 请求数 = 该 position 的平均单 token 能耗。

        判断逻辑:
          - "Prefill step": 并非所有请求都已开始生成 (部分 generated_count == 0)
          - "Decode step":  所有请求都已开始生成, 每个 step 各产出 1 个 token

        Args:
            prompts_token_ids: list of list[int], 每个元素是一条 prompt 的 token IDs
            max_tokens:        每条请求最大生成 token 数
            tracker:           TokenEnergyTracker 实例
            temperature:       采样温度
            top_k:             top-k
            top_p:             top-p

        Returns:
            (results, step_info)
            results: list of dict, 每条请求的结果
            step_info: dict, 包含 prefill/decode 步数和 prefill 能耗
        """
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            top_k=top_k if top_k > 0 else -1,
            top_p=top_p if top_p > 0 else 1.0,
        )

        engine = self._llm.llm_engine
        n_requests = len(prompts_token_ids)

        # ★ 关键: 一次性添加所有请求
        request_ids = []
        for i, token_ids in enumerate(prompts_token_ids):
            req_id = f"batch_{id(self)}_{i}_{time.monotonic_ns()}"
            engine.add_request(
                request_id=req_id,
                prompt={"prompt_token_ids": token_ids},
                params=sampling_params,
            )
            request_ids.append(req_id)

        print(f"[BatchStep] 已添加 {n_requests} 个请求到引擎, "
              f"开始逐 step 生成 (max_tokens={max_tokens})")

        # 跟踪每个请求的状态
        req_state = {
            rid: {"generated_count": 0, "finished": False, "generated_ids": []}
            for rid in request_ids
        }

        # 用 tracker 记录所有 step 的能耗 (包括 prefill 和 decode)
        tracker.new_sequence()

        step_count = 0
        prefill_done = False
        prefill_steps = 0

        while engine.has_unfinished_requests():
            active_count = sum(1 for s in req_state.values() if not s["finished"])
            if active_count == 0:
                break

            # 确定当前 position:
            # 在 prefill 阶段, 用 position = -step_count (负数标记 prefill)
            # 在 decode 阶段, 用 position = decode_step_idx
            if not prefill_done:
                # Prefill: 标记为负数 position
                pos = -(step_count + 1)
            else:
                # Decode: position = step_count - prefill_steps
                pos = step_count - prefill_steps

            # --- 精确测量: begin_token → step() → end_token ---
            tracker.begin_token(pos)
            step_outputs = engine.step()
            tracker.end_token(pos)

            # 更新状态
            for output in step_outputs:
                rid = output.request_id
                if rid not in req_state:
                    continue
                state = req_state[rid]
                if hasattr(output, 'outputs') and output.outputs:
                    cur_count = len(output.outputs[0].token_ids)
                    state["generated_count"] = cur_count
                    state["generated_ids"] = list(output.outputs[0].token_ids)
                if output.finished:
                    state["finished"] = True

            step_count += 1

            # 检查 prefill 是否完成: 所有请求都产出了至少 1 个 token
            if not prefill_done:
                started_count = sum(1 for s in req_state.values()
                                    if s["generated_count"] >= 1)
                if started_count >= n_requests:
                    prefill_done = True
                    prefill_steps = step_count
                    print(f"[BatchStep] Prefill 完成: {prefill_steps} step(s), "
                          f"所有 {n_requests} 个请求已开始生成")

            # 进度
            finished_count = sum(1 for s in req_state.values() if s["finished"])
            decode_pos = step_count - prefill_steps if prefill_done else 0
            if step_count % 100 == 0 or step_count <= 5:
                phase = "decode" if prefill_done else "prefill"
                print(f"    step {step_count} ({phase}, pos={decode_pos}): "
                      f"active={active_count}, "
                      f"finished={finished_count}/{n_requests}")

            # 安全上限
            if step_count >= max_tokens + n_requests + 10:
                print(f"[BatchStep] ⚠ 达到安全上限, 停止")
                break

        decode_steps = step_count - prefill_steps
        print(f"[BatchStep] 完成: prefill={prefill_steps} steps, "
              f"decode={decode_steps} steps, total={step_count} steps")

        # 构造返回结果
        results = []
        for i, (rid, token_ids) in enumerate(zip(request_ids, prompts_token_ids)):
            state = req_state[rid]
            results.append({
                "request_id": rid,
                "prompt_len": len(token_ids),
                "generated_count": state["generated_count"],
                "generated_ids": state["generated_ids"],
            })

        step_info = {
            "total_steps": step_count,
            "prefill_steps": prefill_steps,
            "decode_steps": decode_steps,
        }

        return results, step_info

    def generate_ar_stepwise_stream(
        self,
        prompts_token_ids: list,
        max_tokens: int,
        tracker,
        req_rate: float = 10.0,
        duration: int = 600,
        warmup: int = 0,
        temperature: float = 0.7,
        top_k: int = 10,
        top_p: float = 0.0,
    ):
        """
        vLLM 流式逐 step 生成: 基于时间的请求注入速率模型。

        使用 req_rate (请求/分钟) 和 duration (实验时长/秒) 控制请求注入:
          - 可选的 warmup 阶段: 在计时开始前先注入 warmup 个请求并等它们
            全部完成 prefill (进入 decode 阶段), 确保实验开始时 GPU 已经
            处于稳定的高并发状态
          - 按照 req_rate 速率持续注入请求, 从 prompts_token_ids 中循环取 prompt
          - 达到 duration 后立即结束实验
          - 只记录 decode 阶段的能耗 (跳过 prefill)

        每个 step:
          1. 检查是否该注入新请求 (基于 wall-clock 时间)
          2. 测量整个 step 的 GPU 能耗
          3. 能耗只分摊给处于 decode 阶段的请求 (跳过 prefill)
          4. 将分摊到的能耗归属到每个请求当前的 decode position

        Args:
            prompts_token_ids: list of list[int], prompt pool (循环使用)
            max_tokens:        每条请求最大生成 token 数
            tracker:           TokenEnergyTracker 实例
            req_rate:          每分钟注入的请求数 (默认 10.0 req/min)
            duration:          实验总时长, 秒 (默认 600s = 10min)
            warmup:            预热请求数 (默认 0, 不预热)
            temperature:       采样温度
            top_k:             top-k
            top_p:             top-p

        Returns:
            (results, step_records, stream_info)
            results:      list of dict, 每条请求的结果
            step_records: list of dict, 每个 step 的详细记录
            stream_info:  dict, 实验统计信息
        """
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            top_k=top_k if top_k > 0 else -1,
            top_p=top_p if top_p > 0 else 1.0,
        )

        engine = self._llm.llm_engine
        n_pool = len(prompts_token_ids)  # prompt pool 大小

        # 计算注入间隔 (秒)
        inject_interval_sec = 60.0 / req_rate if req_rate > 0 else float('inf')

        # 请求状态
        req_state = {}
        request_ids = []
        inject_count = 0  # 已注入的请求总数 (含 warmup)

        # 每个请求的 per-position 能耗
        per_request_energies = {}

        # ==================== Warmup 阶段 ====================
        # 在计时开始前先注入 warmup 个请求, 跑完 prefill 进入 decode
        warmup_count = min(warmup, n_pool) if warmup > 0 else 0
        if warmup_count > 0:
            print(f"[StreamStep] ⏳ Warmup: 预注入 {warmup_count} 个请求...")
            for wi in range(warmup_count):
                pool_idx = wi % n_pool
                token_ids = prompts_token_ids[pool_idx]
                req_id = f"stream_{id(self)}_{inject_count}_{time.monotonic_ns()}"
                engine.add_request(
                    request_id=req_id,
                    prompt={"prompt_token_ids": token_ids},
                    params=sampling_params,
                )
                request_ids.append(req_id)
                req_state[req_id] = {
                    "idx": inject_count,
                    "pool_idx": pool_idx,
                    "prompt_len": len(token_ids),
                    "generated_count": 0,
                    "prev_count": 0,
                    "finished": False,
                    "generated_ids": [],
                    "in_prefill": True,
                    "inject_time": 0.0,  # warmup 请求的注入时间记为 0
                    "is_warmup": True,
                }
                per_request_energies[req_id] = []
                inject_count += 1

            # 跑 step 直到所有 warmup 请求都完成 prefill (产出至少 1 个 token)
            warmup_steps = 0
            while True:
                step_outputs = engine.step()
                warmup_steps += 1

                # 更新状态
                for output in step_outputs:
                    rid = output.request_id
                    if rid not in req_state:
                        continue
                    state = req_state[rid]
                    if hasattr(output, 'outputs') and output.outputs:
                        cur_count = len(output.outputs[0].token_ids)
                        state["generated_count"] = cur_count
                        state["generated_ids"] = list(output.outputs[0].token_ids)
                    if output.finished:
                        state["finished"] = True

                # 检查是否所有 warmup 请求都已经产出至少 1 个 token
                all_started = all(
                    s["generated_count"] > 0
                    for rid, s in req_state.items()
                    if s.get("is_warmup", False)
                )
                if all_started:
                    break
                if warmup_steps > 1000:
                    print(f"    [Warmup] ⚠ 超过 1000 步仍未全部完成 prefill, 继续")
                    break

            # 同步 prev_count, 清除 in_prefill 标记
            for rid, state in req_state.items():
                state["prev_count"] = state["generated_count"]
                if state["generated_count"] > 0:
                    state["in_prefill"] = False

            print(f"    [Warmup] ✅ {warmup_count} 个请求已完成 prefill "
                  f"({warmup_steps} steps), 开始正式实验")

        # ==================== 正式实验 ====================
        # 用一个 virtual sequence 来记录所有 step 的能耗
        tracker.new_sequence()

        step_count = 0
        step_records = []

        # 时间控制 (warmup 不算在 duration 内)
        t_start = time.time()
        # 下一次注入的时间点: warmup 之后立即可以注入
        t_next_inject = t_start

        print(f"[StreamStep] 开始流式生成: req_rate={req_rate:.1f} req/min, "
              f"duration={duration}s, max_tokens={max_tokens}")
        print(f"    注入间隔: {inject_interval_sec:.2f}s, "
              f"prompt pool: {n_pool}, warmup: {warmup_count}")

        # 用于追踪正式实验期间新注入的请求数
        experiment_inject_count = 0

        while True:
            t_now = time.time()
            elapsed = t_now - t_start

            # ---- 检查是否达到实验时长 ----
            if elapsed >= duration:
                finished_count = sum(1 for s in req_state.values() if s["finished"])
                unfinished = len(req_state) - finished_count
                print(f"    [StreamStep] ⏱ 达到实验时长 {duration}s, 直接结束实验")
                print(f"    已注入 {inject_count} 个请求 "
                      f"(warmup={warmup_count}, 正式={experiment_inject_count}), "
                      f"已完成 {finished_count}, 未完成 {unfinished}")
                # 中止引擎中所有未完成的请求
                for rid, state in req_state.items():
                    if not state["finished"]:
                        try:
                            engine.abort_request(rid)
                        except Exception:
                            pass
                break

            # ---- 注入新请求 (基于 wall-clock 时间) ----
            if t_now >= t_next_inject:
                # 从 prompt pool 中循环取 prompt
                pool_idx = inject_count % n_pool
                token_ids = prompts_token_ids[pool_idx]
                req_id = f"stream_{id(self)}_{inject_count}_{time.monotonic_ns()}"
                engine.add_request(
                    request_id=req_id,
                    prompt={"prompt_token_ids": token_ids},
                    params=sampling_params,
                )
                request_ids.append(req_id)
                req_state[req_id] = {
                    "idx": inject_count,
                    "pool_idx": pool_idx,
                    "prompt_len": len(token_ids),
                    "generated_count": 0,
                    "prev_count": 0,
                    "finished": False,
                    "generated_ids": [],
                    "in_prefill": True,
                    "inject_time": elapsed,
                    "is_warmup": False,
                }
                per_request_energies[req_id] = []
                inject_count += 1
                experiment_inject_count += 1

                if experiment_inject_count <= 5 or experiment_inject_count % 50 == 0:
                    print(f"    [StreamStep] 注入请求 #{experiment_inject_count} "
                          f"(pool={pool_idx}, prompt_len={len(token_ids)}, "
                          f"t={elapsed:.1f}s)")

                t_next_inject = t_start + experiment_inject_count * inject_interval_sec

            # ---- 检查是否还有未完成的请求 ----
            if not engine.has_unfinished_requests():
                # 引擎空了但还没到时间, 短暂等待下一次注入
                time.sleep(min(0.001, inject_interval_sec * 0.1))
                continue

            # ---- 执行一个 step, 测量能耗 ----
            tracker.begin_token(step_count)
            step_outputs = engine.step()
            tracker.end_token(step_count)

            # 获取本 step 的能耗
            step_energy = 0.0
            if tracker._current_seq:
                last_entry = tracker._current_seq[-1]
                step_energy = last_entry[1]

            # ---- 更新请求状态 ----
            for output in step_outputs:
                rid = output.request_id
                if rid not in req_state:
                    continue
                state = req_state[rid]
                state["prev_count"] = state["generated_count"]
                if hasattr(output, 'outputs') and output.outputs:
                    cur_count = len(output.outputs[0].token_ids)
                    state["generated_count"] = cur_count
                    state["generated_ids"] = list(output.outputs[0].token_ids)
                if output.finished:
                    state["finished"] = True

            # ---- 识别本 step 中哪些请求产出了新 token ----
            # 对于不在 step_outputs 中的请求, 同步 prev_count 防止重复计入
            output_rids = {o.request_id for o in step_outputs
                          if hasattr(o, 'request_id')}
            for rid, state in req_state.items():
                if rid not in output_rids:
                    state["prev_count"] = state["generated_count"]

            active_positions = {}
            prefill_count_this_step = 0
            for rid, state in req_state.items():
                if state["finished"] and state["generated_count"] == state["prev_count"]:
                    continue
                new_tokens = state["generated_count"] - state["prev_count"]
                if new_tokens > 0:
                    if state["in_prefill"]:
                        # 该请求刚完成 prefill, 标记为 decode
                        state["in_prefill"] = False
                        prefill_count_this_step += 1
                        # ★ 跳过 prefill: 不记录 position 0 的能耗
                        continue
                    decode_pos = state["generated_count"] - 1
                    active_positions[rid] = decode_pos

            num_decode_active = len(active_positions)

            # ---- 将能耗均分到 decode 阶段的请求 (跳过 prefill) ----
            if num_decode_active > 0 and step_energy > 0:
                per_req_energy = step_energy / num_decode_active
                for rid, decode_pos in active_positions.items():
                    per_request_energies[rid].append((decode_pos, per_req_energy))

            # ---- 记录 step 信息 ----
            step_records.append({
                "step": step_count,
                "time": elapsed,
                "step_energy_mj": step_energy,
                "num_active": num_decode_active,
                "num_prefill": prefill_count_this_step,
            })

            step_count += 1

            # 进度
            finished_count = sum(1 for s in req_state.values() if s["finished"])
            injected = len(req_state)
            unfinished_count = injected - finished_count
            if step_count % 500 == 0 or step_count <= 5:
                print(f"    step {step_count}: t={elapsed:.0f}s, "
                      f"injected={injected} (warmup={warmup_count}), "
                      f"decode_active={num_decode_active}, "
                      f"prefill={prefill_count_this_step}, "
                      f"finished={finished_count}")

            # 安全上限
            if step_count >= 500000:
                print(f"[StreamStep] ⚠ 达到安全上限 {step_count}, 停止")
                break

        total_time = time.time() - t_start
        total_generated = sum(s["generated_count"] for s in req_state.values())
        finished_count = sum(1 for s in req_state.values() if s["finished"])

        print(f"[StreamStep] 完成: {step_count} steps in {total_time:.1f}s, "
              f"{inject_count} requests total (warmup={warmup_count}), "
              f"{finished_count} finished, "
              f"{total_generated} tokens "
              f"({total_generated/total_time:.1f} tok/s)")

        # ---- 构造返回结果 ----
        results = []
        for rid in request_ids:
            state = req_state[rid]
            results.append({
                "request_id": rid,
                "idx": state["idx"],
                "pool_idx": state["pool_idx"],
                "prompt_len": state["prompt_len"],
                "generated_count": state["generated_count"],
                "generated_ids": state["generated_ids"],
                "per_position_energies": per_request_energies[rid],
                "inject_time": state["inject_time"],
                "is_warmup": state.get("is_warmup", False),
            })

        stream_info = {
            "total_steps": step_count,
            "total_time_s": total_time,
            "total_injected": inject_count,
            "warmup_count": warmup_count,
            "experiment_injected": experiment_inject_count,
            "total_finished": finished_count,
            "total_generated": total_generated,
            "actual_req_rate": experiment_inject_count / (min(total_time, duration) / 60)
                               if total_time > 0 else 0,
        }

        return results, step_records, stream_info

    # --- DSD / DSSD draft 接口 (vLLM 不支持逐 token 控制, 回退到 PyTorch 行为) ---
    # 注: vLLM 主要用于 local_baseline 的高性能自回归,
    #     DSD/DSSD 的 draft 仍需逐 token 生成, 应使用 PyTorchDraftNode

    def draft_step_DSD(self, prefix: torch.Tensor, gamma: int):
        raise NotImplementedError(
            "VLLMDraftNode 不支持 DSD draft (需要逐 token 控制)。"
            "DSD/DSSD 模式请使用 PyTorch 引擎 (--engine pytorch)。"
        )

    def draft_step_DSSD(self, prefix: torch.Tensor, gamma: int):
        raise NotImplementedError(
            "VLLMDraftNode 不支持 DSSD draft (需要逐 token 控制)。"
            "DSD/DSSD 模式请使用 PyTorch 引擎 (--engine pytorch)。"
        )

    def resample_DSSD(self, j, pj, q_probs):
        raise NotImplementedError(
            "VLLMDraftNode 不支持 DSSD resample。"
        )


# ============ MLX 后端 ============

class MLXDraftNode:
    """MLX 后端的 UAV draft 节点（Apple Silicon 原生加速）"""

    def __init__(self, model, tokenizer_unused, args):
        import mlx.core as mx
        self.mx = mx
        self.framework = "mlx"
        self.model = model
        self.args = args
        self.device = torch.device("cpu")
        print(f"[UAV] Draft model loaded on: Apple Silicon MLX (native Metal acceleration)")

    def _mlx_sample(self, logits_1d):
        """
        从 MLX logits 中采样一个 token（带 temperature / top_k / top_p）
        返回: token_id (int), probs (mx.array, shape (V,))
        """
        mx = self.mx
        temperature = self.args.temperature
        top_k = self.args.top_k
        top_p = self.args.top_p

        logits = logits_1d / temperature

        if top_k > 0:
            k = min(top_k, logits.shape[-1])
            top_values = mx.sort(logits)[-k:]
            threshold = top_values[0]
            logits = mx.where(logits < threshold, mx.array(float('-inf')), logits)

        if top_p > 0.0:
            sorted_indices = mx.argsort(logits)[::-1]
            sorted_logits = logits[sorted_indices]
            sorted_probs = mx.softmax(sorted_logits)
            cumulative_probs = mx.cumsum(sorted_probs)
            mask = cumulative_probs - sorted_probs > top_p
            sorted_logits = mx.where(mask, mx.array(float('-inf')), sorted_logits)
            logits = mx.zeros_like(logits)
            logits[sorted_indices] = sorted_logits

        probs = mx.softmax(logits)
        token_id = mx.random.categorical(logits).item()

        if token_id == 0:
            token_id = mx.argmax(probs).item()

        return token_id, probs

    def draft_step_DSD(self, prefix: torch.Tensor, gamma: int):
        """DSD Draft (MLX): 返回 torch.Tensor"""
        mx = self.mx
        x_np = prefix.cpu().numpy()
        x_mx = mx.array(x_np)
        q_stack = []

        for _ in range(gamma):
            logits_all = self.model(x_mx)
            last_logits = logits_all[0, -1, :]
            q_stack.append(np.array(last_logits.astype(mx.float32)))
            tok_id, _ = self._mlx_sample(last_logits)
            next_tok = mx.array([[tok_id]])
            x_mx = mx.concatenate([x_mx, next_tok], axis=1)

        x_draft = torch.from_numpy(np.array(x_mx.astype(mx.int64))).long()
        q_step_logits = torch.from_numpy(np.stack(q_stack, axis=0)).float()
        raw_bytes = tensor_nbytes(q_step_logits)
        return x_draft, q_step_logits, raw_bytes

    def draft_step_DSSD(self, prefix: torch.Tensor, gamma: int):
        """DSSD Draft (MLX): 返回 torch.Tensor"""
        mx = self.mx
        x_np = prefix.cpu().numpy()
        x_mx = mx.array(x_np)
        q_probs_list = []
        q_values = []

        for _ in range(gamma):
            logits_all = self.model(x_mx)
            last_logits = logits_all[0, -1, :]
            q_dist = mx.softmax(last_logits)
            q_dist_np = np.array(q_dist.astype(mx.float32))
            q_probs_list.append(q_dist_np)

            tok_id, _ = self._mlx_sample(last_logits)
            next_tok = mx.array([[tok_id]])
            x_mx = mx.concatenate([x_mx, next_tok], axis=1)
            q_values.append(float(q_dist_np[tok_id]))

        x_draft = torch.from_numpy(np.array(x_mx.astype(mx.int64))).long()
        q_probs = torch.from_numpy(np.stack(q_probs_list, axis=0)).float()

        token_bytes = x_draft.numel() * 4
        prob_bytes = len(q_values) * 4
        dup_bytes = token_bytes + prob_bytes
        return x_draft, q_values, q_probs, dup_bytes

    def resample_DSSD(self, j: int, pj: torch.Tensor, q_probs: torch.Tensor) -> torch.Tensor:
        """DSSD Resample"""
        q_j = q_probs[j - 1]
        diff = (pj - q_j).clamp(min=0)
        diff = diff / diff.sum()
        xj_prime = torch.multinomial(diff.cpu().float(), 1).unsqueeze(0)
        return xj_prime


# ============ 工厂函数 ============

def _get_gpu_compute_capability_no_init() -> tuple:
    """
    获取 GPU compute capability, **不初始化 CUDA 上下文**。

    vLLM 多卡模式需要 fork 子进程, 如果主进程已经初始化了 CUDA,
    子进程会报 "Cannot re-initialize CUDA in forked subprocess"。
    因此这里用 pynvml (纯 C 库调用) 而非 torch.cuda 来获取 GPU 信息。

    Returns:
        (major, minor) 或 (0, 0) 如果无法获取
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        return (major, minor)
    except Exception:
        pass

    # 退回方案: 用 nvidia-smi 查询
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            line = result.stdout.strip().split("\n")[0].strip()
            parts = line.split(".")
            if len(parts) == 2:
                return (int(parts[0]), int(parts[1]))
    except Exception:
        pass

    return (0, 0)


def _has_nvidia_gpu_no_init() -> bool:
    """
    检查是否有 NVIDIA GPU, **不初始化 CUDA 上下文**。

    使用 pynvml 而非 torch.cuda.is_available() 来避免提前初始化 CUDA。
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        return count > 0
    except Exception:
        pass

    # 退回: 检查 nvidia-smi 是否存在
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0 and "GPU" in result.stdout
    except Exception:
        return False


def _is_model_vllm_compatible(model_name: str) -> tuple:
    """
    检查模型是否与 vLLM 兼容 (特别是在 V100 等老 GPU 上)。

    **重要**: 此函数不调用任何 torch.cuda API, 避免提前初始化 CUDA,
    否则 vLLM 多卡 fork 会报 "Cannot re-initialize CUDA in forked subprocess"。

    已知不兼容的情况:
      - Qwen3.5 系列: 多模态模型 (Qwen3_5ForConditionalGeneration),
        在 V100 (compute capability 7.0) 上 vLLM 会卡在 encoder cache profiling。
      - 其他 ConditionalGeneration 多模态模型在 V100 上也可能有类似问题。

    Returns:
        (compatible: bool, reason: str)
    """
    import json, os

    # 直接读取 config.json, 避免 AutoConfig 不认识新模型类型的问题
    config_dict = {}
    config_path = os.path.join(model_name, "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r") as f:
                config_dict = json.load(f)
        except Exception:
            pass

    if not config_dict:
        # 尝试用 AutoConfig (对于 HuggingFace Hub 上的模型)
        try:
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            config_dict = cfg.to_dict()
        except Exception:
            return True, ""  # 无法读取 config, 不做限制

    # 获取模型架构
    archs = config_dict.get("architectures", []) or []
    model_type = config_dict.get("model_type", "")

    # 检查是否是多模态 / ConditionalGeneration 模型
    is_multimodal = any("ConditionalGeneration" in a for a in archs)
    if not is_multimodal:
        # 检查是否有 vision / image 相关配置
        is_multimodal = (
            "vision_config" in config_dict or
            "image_token_id" in config_dict or
            "visual" in config_dict
        )

    if not is_multimodal:
        return True, ""

    # 多模态模型: 检查 GPU compute capability (不初始化 CUDA!)
    cc = _get_gpu_compute_capability_no_init()
    if cc[0] > 0 and cc[0] < 8:
        reason = (f"多模态模型 ({archs[0] if archs else model_type}) "
                  f"在 GPU compute capability {cc[0]}.{cc[1]} (< 8.0) 上 "
                  f"vLLM 会卡在 encoder cache profiling, 自动回退 PyTorch")
        return False, reason

    return True, ""


def _should_use_vllm(engine: str, device_str: str, model_name: str = "") -> bool:
    """
    判断是否应该使用 vLLM 引擎。

    **重要**: 此函数不调用任何 torch.cuda API, 避免提前初始化 CUDA。
    """
    if engine == "pytorch":
        return False
    if engine == "vllm":
        # 即使强制 vLLM, 也检查兼容性并给出警告
        if model_name:
            ok, reason = _is_model_vllm_compatible(model_name)
            if not ok:
                print(f"[UAV] ⚠ 警告: {reason}")
                print(f"[UAV] ⚠ 你强制指定了 --engine vllm, 继续尝试...")
        return True
    # engine == "auto": 有 NVIDIA GPU 且 vLLM 可用时使用
    # 用 pynvml 检测, 不初始化 CUDA 上下文
    if not _has_nvidia_gpu_no_init():
        return False
    try:
        import vllm  # noqa: F401
    except ImportError:
        return False

    # 检查模型是否与 vLLM 兼容
    if model_name:
        ok, reason = _is_model_vllm_compatible(model_name)
        if not ok:
            print(f"[UAV] {reason}")
            return False

    return True


def create_draft_node(model_name: str, device_str: str, framework: str, args,
                      gpu_ids: str = None, engine: str = "auto"):
    """
    工厂函数: 根据框架和引擎选择创建对应的 DraftNode
    返回: (draft_node, tokenizer)

    引擎选择:
      - engine="auto"    → 有 NVIDIA GPU 且 vLLM 可用 且模型兼容时用 vLLM, 否则 PyTorch
      - engine="vllm"    → 强制使用 vLLM (不兼容时给出警告)
      - engine="pytorch" → 强制使用 PyTorch

    多卡支持:
      - device="auto"              → 自动分配到所有可用 GPU
      - device="auto" + gpu_ids="0,1,2"  → 分配到指定 GPU
    """
    fw = detect_framework(device_str, framework)
    print(f"[UAV] Selected framework: {fw.upper()}")

    if fw == "mlx":
        from mlx_lm import load as mlx_load
        print(f"[UAV Client] Loading draft model (MLX): {model_name}")
        mlx_model, tokenizer = mlx_load(model_name)
        node = MLXDraftNode(mlx_model, None, args)
        return node, tokenizer

    # ---- 检查是否使用 vLLM (传入 model_name 做兼容性检查) ----
    use_vllm = _should_use_vllm(engine, device_str, model_name)

    if use_vllm:
        return _create_vllm_node(model_name, args, gpu_ids)

    # ---- PyTorch 后端 ----
    return _create_pytorch_node(model_name, device_str, args, gpu_ids)


def _get_gpu_count_no_init() -> int:
    """获取 GPU 数量, 不初始化 CUDA 上下文。"""
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetCount()
    except Exception:
        pass
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
            return len(lines)
    except Exception:
        pass
    return 1


def _create_vllm_node(model_name: str, args, gpu_ids: str = None):
    """
    创建 vLLM 引擎节点。

    **重要**: 在调用 vllm.LLM() 之前, 不能有任何 torch.cuda API 调用,
    否则 vLLM 多卡 fork 会报 "Cannot re-initialize CUDA in forked subprocess"。
    """
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 确定 tensor_parallel_size (不初始化 CUDA!)
    if gpu_ids:
        gpu_list = [int(x.strip()) for x in gpu_ids.split(",")]
        tp_size = len(gpu_list)
    else:
        tp_size = _get_gpu_count_no_init()

    # V100 (cc < 8.0) 不支持 bfloat16, 需要强制使用 float16
    # 同时 V100 上 Triton 3.x JIT 编译某些 kernel 会崩溃
    # (LLVM ERROR: Failed to compute parent layout for slice layout),
    # 需要:
    #   1. enforce_eager=True — 禁用 CUDA graph
    #   2. compilation_config={"level": 0} — 禁用 torch.compile (NO_COMPILATION)
    cc = _get_gpu_compute_capability_no_init()
    vllm_extra_kwargs = {}
    if cc[0] > 0 and cc[0] < 8:
        vllm_dtype = "half"
        vllm_extra_kwargs["enforce_eager"] = True
        vllm_extra_kwargs["compilation_config"] = {"level": 0}
        vllm_extra_kwargs["enable_chunked_prefill"] = False
        print(f"[UAV Client] GPU compute capability {cc[0]}.{cc[1]} < 8.0, "
              f"使用 dtype=half, enforce_eager=True, compilation_level=0 (NO_COMPILATION)")
    else:
        vllm_dtype = "auto"

    print(f"[UAV Client] Loading draft model with vLLM engine: {model_name}")
    print(f"[UAV Client] tensor_parallel_size = {tp_size}, dtype = {vllm_dtype}")

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        dtype=vllm_dtype,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        **vllm_extra_kwargs,
    )

    node = VLLMDraftNode(llm, tokenizer, args)
    return node, tokenizer


def _create_pytorch_node(model_name: str, device_str: str, args,
                         gpu_ids: str = None):
    """创建 PyTorch 后端节点"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device_str_lower = device_str.strip().lower()

    # ---- 多卡: device="auto" 使用 device_map ----
    if device_str_lower == "auto" and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"[UAV Client] Loading draft model (PyTorch multi-GPU): {model_name}")
        load_kwargs = {"torch_dtype": "auto", "device_map": "auto"}
        if gpu_ids:
            gpu_list = [int(x.strip()) for x in gpu_ids.split(",")]
            max_memory = {i: "80GiB" for i in gpu_list}
            load_kwargs["max_memory"] = max_memory
            print(f"[UAV Client] Using GPUs: {gpu_list}")
        draft_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        # 确定输入设备 (模型第一层所在的 GPU)
        if hasattr(draft_model, "hf_device_map"):
            first_dev = next(iter(draft_model.hf_device_map.values()))
            if isinstance(first_dev, int):
                input_device = torch.device(f"cuda:{first_dev}")
            elif isinstance(first_dev, str) and first_dev.startswith("cuda"):
                input_device = torch.device(first_dev)
            else:
                input_device = torch.device(first_dev)
        else:
            input_device = torch.device("cuda:0")

        node = PyTorchDraftNode(draft_model, input_device, args)
        return node, tokenizer

    # ---- 单卡 / CPU / MPS ----
    device = resolve_device(device_str)
    print(f"[UAV Client] Loading draft model (PyTorch): {model_name}")
    draft_model = AutoModelForCausalLM.from_pretrained(model_name)
    node = PyTorchDraftNode(draft_model, device, args)
    return node, tokenizer
