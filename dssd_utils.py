"""
dssd_utils.py - 共享工具函数（UAV端和BS端都需要）

包含: 设备检测、采样函数、logits压缩、tensor工具
（能耗监控已移至 energy_monitor.py）
"""

import torch
import torch.nn.functional as F


# ============ 设备自动检测 ============

def resolve_device(device_str: str) -> torch.device:
    """
    智能解析设备字符串，自动适配不同硬件:
      - "auto"       → 自动检测最佳设备 (cuda > mps > cpu)
      - "cuda"       → 使用第一块 CUDA GPU
      - "cuda:0"     → 使用指定 CUDA GPU
      - "mps"        → 使用 Apple Silicon GPU
      - "cpu"        → 使用 CPU
    """
    device_str = device_str.strip().lower()

    if device_str == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda:0")
            print(f"[Device] Auto-detected CUDA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
            print(f"[Device] Auto-detected Apple MPS (Metal Performance Shaders)")
        else:
            dev = torch.device("cpu")
            print(f"[Device] Auto-detected CPU")
        return dev

    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"[Device] WARNING: CUDA not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device(device_str)

    if device_str == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print(f"[Device] WARNING: MPS not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("mps")

    return torch.device(device_str)


def get_device_info(device: torch.device) -> str:
    """获取设备的详细信息字符串"""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        mem = torch.cuda.get_device_properties(device).total_mem / (1024**3)
        return f"CUDA:{device.index} ({name}, {mem:.1f}GB)"
    elif device.type == "mps":
        return "Apple MPS (Metal Performance Shaders)"
    else:
        return "CPU"


# ============ 采样相关 ============

def top_k_top_p_filter(logits, top_k: int = 0, top_p: float = 0.0):
    if top_k > 0:
        filter_val = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter_val[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        filter_mask = cumulative_probs > top_p
        filter_mask[..., 1:] = filter_mask[..., :-1].clone()
        filter_mask[..., 0] = 0
        indices_to_remove = filter_mask.scatter(1, sorted_indices, filter_mask)
        logits[indices_to_remove] = float('-inf')
    return logits


def sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """
    从 logits 中采样下一个 token（兼容 CUDA / MPS / CPU）
    Args:
        logits: shape (batch, vocab)
    Returns:
        next token with shape (batch, 1)
    """
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=1)

    # MPS 后端的 torch.multinomial 在某些 PyTorch 版本中可能有 bug，
    # 回退到 CPU 采样后再移回原设备
    src_device = probs.device
    if src_device.type == "mps":
        idx_next = torch.multinomial(probs.cpu(), num_samples=1).to(src_device)
    else:
        idx_next = torch.multinomial(probs, num_samples=1)

    if idx_next.item() == 0:
        raise RuntimeError("Sampled token id is 0 (usually <pad>)")
    return idx_next


# ============ 通信相关工具 ============

def tensor_nbytes(t: torch.Tensor) -> int:
    """计算 tensor 占用的字节数"""
    return t.element_size() * t.numel()


def compress_logits(logits: torch.Tensor, k: int = 8) -> torch.Tensor:
    """logits: (V,) → 返回 top-k 压缩后的张量"""
    top = torch.topk(logits, k=k)
    ids = top.indices.to(torch.int32)
    prob = torch.softmax(top.values, dim=-1).to(torch.float16)
    ids_bytes = ids.view(torch.uint8)
    prob_bytes = prob.view(torch.uint8)
    return torch.cat([ids_bytes, prob_bytes])
