"""
bs_server.py - 基站端 (BS) 服务器
在服务器/基站设备上运行，加载大模型，提供 verify 服务

用法（在 BS 机器上执行）:

  单卡加载:
    python bs_server.py \
        --target_model_name ./LLM/opt-1.3b \
        --device cuda:0 \
        --port 50051

  多卡自动切分（大模型放不下一张卡时）:
    python bs_server.py \
        --target_model_name ./LLM/opt-13b \
        --device auto \
        --port 50051

  指定多卡分配（手动控制哪些卡参与）:
    python bs_server.py \
        --target_model_name ./LLM/opt-13b \
        --device auto \
        --gpu_ids 0,1,2 \
        --port 50051

  CPU offload（显存不够时把一部分层放到 CPU）:
    python bs_server.py \
        --target_model_name ./LLM/opt-30b \
        --device auto \
        --cpu_offload \
        --port 50051

依赖: torch, transformers, accelerate（多卡时需要）
"""

import argparse
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from dssd_net import BSServer
from dssd_utils import sample, tensor_nbytes, resolve_device, get_device_info


def load_target_model(model_name: str, device_str: str, gpu_ids: str = None,
                      cpu_offload: bool = False, dtype_str: str = "auto"):
    """
    加载大模型，支持多种部署方式:
    1. 单卡:          device="cuda:0"
    2. 多卡自动切分:   device="auto"
    3. 指定多卡:       device="auto" + gpu_ids="0,1,2"
    4. CPU offload:    device="auto" + cpu_offload=True
    5. 纯 CPU:        device="cpu"

    返回: (model, device_for_input)
      - model: 加载好的模型
      - device_for_input: 输入 tensor 应该放到的设备
        (多卡时为第一块 GPU，单卡时为指定 GPU)
    """
    # 解析 dtype
    dtype_map = {
        "auto": "auto",
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype_str, "auto")

    device_str = device_str.strip().lower()

    # ---- 情况 1: 多卡/自动分配 (device_map) ----
    if device_str == "auto":
        print(f"[BS] Loading model with device_map='auto' (multi-GPU / auto-split)")

        # 构建 device_map 参数
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": "auto",
        }

        # 如果指定了 GPU 列表，限制可用设备
        if gpu_ids:
            gpu_list = [int(x.strip()) for x in gpu_ids.split(",")]
            max_memory = {i: "80GiB" for i in gpu_list}
            if cpu_offload:
                max_memory["cpu"] = "64GiB"
            load_kwargs["max_memory"] = max_memory
            print(f"[BS] Using GPUs: {gpu_list}" +
                  (", with CPU offload" if cpu_offload else ""))
        elif cpu_offload:
            load_kwargs["device_map"] = "auto"
            # accelerate 会自动分配 CPU offload
            print(f"[BS] CPU offload enabled")

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        model.eval()

        # 确定输入设备: 多卡时输入应放到模型的第一层所在的设备
        if hasattr(model, "hf_device_map"):
            first_device = next(iter(model.hf_device_map.values()))
            if isinstance(first_device, int):
                input_device = torch.device(f"cuda:{first_device}")
            elif isinstance(first_device, str) and first_device.startswith("cuda"):
                input_device = torch.device(first_device)
            else:
                input_device = torch.device(first_device)
            print(f"[BS] Model distributed across: {set(model.hf_device_map.values())}")
            print(f"[BS] Input device: {input_device}")
        else:
            input_device = torch.device("cuda:0")

        return model, input_device

    # ---- 情况 2: 单设备 (cuda:X / cpu) ----
    device = resolve_device(device_str)
    print(f"[BS] Loading model on single device: {get_device_info(device)}")

    load_kwargs = {}
    if torch_dtype != "auto":
        load_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model = model.to(device)
    model.eval()

    return model, device


class BSVerifier:
    """基站端验证逻辑"""

    def __init__(self, model, input_device, tokenizer, verbose=False):
        """
        Args:
            model: 加载好的大模型（可能跨多卡）
            input_device: 输入 tensor 应放到的设备
            tokenizer: tokenizer
            verbose: 是否打印详细日志
        """
        self.model = model
        self.input_device = input_device
        self.tokenizer = tokenizer
        self.verbose = verbose
        print(f"[BS] Verifier ready, input device: {input_device}")

    def handle_request(self, request: dict) -> dict:
        """统一请求分发"""
        method = request.get("method")
        if method == "verify_dsd":
            return self._verify_dsd(request)
        elif method == "verify_dssd":
            return self._verify_dssd(request)
        elif method == "autoregressive":
            return self._autoregressive(request)
        else:
            return {"error": f"Unknown method: {method}"}

    def _verify_dsd(self, req: dict) -> dict:
        """DSD: 经典分布式投机解码验证"""
        t_start = time.time()

        x_draft = req["x_draft"]
        q_steps = req["q_probs"]
        gamma = req["gamma"]
        temperature = req["temperature"]

        correct_num = reject_num = 0
        prefix_len = x_draft.size(1) - gamma

        with torch.no_grad():
            p_all = self.model(x_draft.to(self.input_device)).logits.cpu()
        p_slice = p_all[0, prefix_len - 1: prefix_len + gamma - 1, :]

        p_probs = F.softmax(p_slice / temperature, dim=-1)
        q_probs = F.softmax(q_steps / temperature, dim=-1)

        n = prefix_len + gamma - 1
        t_corr = None
        for i in range(gamma):
            tok_id = int(x_draft[0, prefix_len + i].item())
            if torch.rand(1).item() > (p_probs[i, tok_id] / q_probs[i, tok_id]):
                n = prefix_len + i - 1
                diff = (p_probs[i] - q_probs[i]).clamp(min=0)
                diff = diff / diff.sum()
                t_corr = torch.multinomial(diff, 1).unsqueeze(0)
                reject_num += 1
                break
            else:
                correct_num += 1

        if t_corr is None:
            prob_last = F.softmax(p_all[0, n] / temperature, dim=-1)
            t_corr = torch.multinomial(prob_last, 1).unsqueeze(0)

        elapsed = time.time() - t_start
        if self.verbose:
            print(f"  [DSD verify] accept={correct_num} reject={reject_num} time={elapsed:.4f}s")

        return {
            "n": n,
            "t_corr": t_corr,
            "correct_num": correct_num,
            "reject_num": reject_num,
            "bs_compute_time": elapsed,
        }

    def _verify_dssd(self, req: dict) -> dict:
        """DSSD: 分布式拆分投机解码验证（仅用概率值）"""
        t_start = time.time()

        x_draft = req["x_draft"]
        q_values = req["q_values"]
        gamma = req["gamma"]
        temperature = req["temperature"]

        correct_num = 0
        reject_num = 0
        prefix_len = x_draft.size(1) - gamma

        with torch.no_grad():
            p_all = self.model(x_draft.to(self.input_device)).logits.cpu()
        p_logits = p_all[0, prefix_len - 1: prefix_len + gamma, :]
        p_probs = F.softmax(p_logits / temperature, dim=-1)

        flag = 1
        j = 1
        pj = None
        xj = None

        for i in range(gamma):
            current_j = i + 1
            tok_id = int(x_draft[0, prefix_len + i].item())
            q_i = q_values[i]
            p_i = p_probs[i, tok_id].item()

            if torch.rand(1).item() > min(1.0, p_i / q_i):
                flag = 0
                j = current_j
                pj = p_probs[i]
                reject_num += 1
                break
            else:
                correct_num += 1

        if flag == 1:
            j = gamma + 1
            xj = torch.multinomial(p_probs[gamma], 1).unsqueeze(0)

        elapsed = time.time() - t_start
        if self.verbose:
            status = "ALL ACCEPT" if flag == 1 else f"REJECT@{j}"
            print(f"  [DSSD verify] {status} accept={correct_num} reject={reject_num} time={elapsed:.4f}s")

        return {
            "j": j,
            "flag": flag,
            "pj": pj,       # tensor or None
            "xj": xj,       # tensor or None
            "correct_num": correct_num,
            "reject_num": reject_num,
            "bs_compute_time": elapsed,
        }

    def _autoregressive(self, req: dict) -> dict:
        """Baseline: 大模型纯自回归生成"""
        input_ids = req["input_ids"].to(self.input_device)
        max_len = req["max_len"]
        temperature = req["temperature"]
        top_k = req["top_k"]
        top_p = req["top_p"]

        prefix = input_ids
        n = prefix.shape[1]
        T = n + max_len

        t_start = time.time()
        with torch.no_grad():
            while n < T:
                logits = self.model(prefix).logits[:, -1, :]
                # 多卡时 logits 可能在最后一层的设备上，移到 CPU 采样
                logits_cpu = logits.cpu()
                idx_next = sample(logits_cpu, temperature, top_k, top_p)
                prefix = torch.cat((prefix, idx_next.to(self.input_device)), dim=1)
                n += 1

        elapsed = time.time() - t_start
        throughput = max_len / elapsed if elapsed > 0 else 0
        print(f"  [Autoregressive] {throughput:.2f} tokens/s, time={elapsed:.2f}s")

        return {
            "output_ids": prefix.cpu(),
            "time_cost": elapsed,
            "throughput": throughput,
        }


def main():
    parser = argparse.ArgumentParser(description="BS (Base Station) Server")
    parser.add_argument('--target_model_name', type=str, default="./LLM/opt-1.3b",
                        help="Path to the large (target) model")
    parser.add_argument('--device', type=str, default="auto",
                        help="Device: 'auto' (multi-GPU auto-split), 'cuda:0', 'cpu', etc.")
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help="Comma-separated GPU IDs to use, e.g. '0,1,2' (only with --device auto)")
    parser.add_argument('--cpu_offload', action='store_true',
                        help="Enable CPU offload when GPU memory is insufficient (only with --device auto)")
    parser.add_argument('--dtype', type=str, default="auto",
                        choices=["auto", "fp32", "fp16", "bf16"],
                        help="Model precision: auto, fp32, fp16, bf16")
    parser.add_argument('--port', type=int, default=50051,
                        help="TCP server port")
    parser.add_argument('--verbose', action='store_true',
                        help="Print detailed logs")
    args = parser.parse_args()

    # 打印 GPU 信息
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"[BS Server] Found {n_gpus} GPU(s):")
        for i in range(n_gpus):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_mem / (1024**3)
            print(f"  GPU {i}: {name} ({mem:.1f} GB)")
    else:
        print("[BS Server] No CUDA GPUs found, will use CPU")

    # 加载模型
    print(f"\n[BS Server] Loading model: {args.target_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_name)
    model, input_device = load_target_model(
        model_name=args.target_model_name,
        device_str=args.device,
        gpu_ids=args.gpu_ids,
        cpu_offload=args.cpu_offload,
        dtype_str=args.dtype,
    )

    verifier = BSVerifier(model, input_device, tokenizer, verbose=args.verbose)

    server = BSServer(host="0.0.0.0", port=args.port)
    server.start(handler_fn=verifier.handle_request)


if __name__ == "__main__":
    main()
