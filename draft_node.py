"""
draft_node.py - UAV 端 Draft 节点（小模型推理）

包含:
  - PyTorchDraftNode: PyTorch 后端 (CUDA / MPS / CPU)
  - MLXDraftNode:     MLX 后端 (Apple Silicon 原生加速)
  - detect_framework(): 自动检测推理框架
  - create_draft_node(): 工厂函数
"""

import platform
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    if device_str.startswith("cuda") or (device_str == "auto" and torch.cuda.is_available()):
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
    """PyTorch 后端的 UAV draft 节点（支持 CUDA / MPS / CPU）"""

    def __init__(self, model, device: torch.device, args):
        self.device = device
        self.framework = "pytorch"
        self.model = model.to(device)
        self.model.eval()
        self.args = args
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

def create_draft_node(model_name: str, device_str: str, framework: str, args):
    """
    工厂函数: 根据框架选择创建对应的 DraftNode
    返回: (draft_node, tokenizer)
    """
    fw = detect_framework(device_str, framework)
    print(f"[UAV] Selected framework: {fw.upper()}")

    if fw == "mlx":
        from mlx_lm import load as mlx_load
        print(f"[UAV Client] Loading draft model (MLX): {model_name}")
        mlx_model, tokenizer = mlx_load(model_name)
        node = MLXDraftNode(mlx_model, None, args)
        return node, tokenizer

    else:  # pytorch
        device = resolve_device(device_str)
        print(f"[UAV Client] Loading draft model (PyTorch): {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        draft_model = AutoModelForCausalLM.from_pretrained(model_name)
        node = PyTorchDraftNode(draft_model, device, args)
        return node, tokenizer
