"""
energy_monitor.py - UAV 端能耗监控模块

包含:
  - EnergyMonitor: 进程级能耗监控器（计算 + 存储 + 网络 分离）
  - TokenEnergyTracker: 逐 token 能耗记录器 (基于 Zeus 硬件计数器)
  - 设备 TDP 常量表
  - GPU 能耗分析模型（计算 vs 显存搬运）
  - Zeus 硬件能耗计数器集成 (优先使用)
  - 网络功耗 / 吞吐量常量表

能耗测量优先级:
  1. Zeus (硬件能耗计数器, nvmlDeviceGetTotalEnergyConsumption)
     - 总能耗: 最准确, 基于 GPU 内部硬件计数器
     - H100+ GPU: 可直接测量 HBM 内存功率 (NVML_POWER_SCOPE_MEMORY)
     - V100/A100: 用分析模型 (Analytical Model) 拆分 compute/memory
  2. pynvml 采样 (退化方案, Zeus 不可用时使用)
     - 周期性采样 GPU 功率, 积分得总能耗
  3. TDP 估算 (最后退化方案)
     - wall_time × TDP × load_factor

Token-level 能耗记录:
  TokenEnergyTracker 利用 Zeus 的 begin_window/end_window 机制,
  在每个 token 生成前后各调用一次, 得到该 token 的精确能耗 (mJ)。
  最终按 token 位置取平均, 可视化能耗随位置的变化趋势。
  适用于 PyTorch 逐 token 生成和 vLLM 0.8.1 的同步 step() 调用。
"""

import time
import resource
import platform
import threading
import subprocess
import warnings
from typing import Dict, List, Optional

import torch


# ============ 常量表 ============

# 网络接口典型功耗 (mW)，用于传输能耗估算
# 参考: IEEE 802.11 标准实测数据
_NET_POWER_MW = {
    "wifi_tx":   800,    # WiFi 发送 ~800 mW (802.11ac/ax)
    "wifi_rx":   500,    # WiFi 接收 ~500 mW
    "wifi_idle": 50,     # WiFi 空闲 ~50 mW
    "lte_tx":    1200,   # LTE 发送 ~1200 mW
    "lte_rx":    800,    # LTE 接收 ~800 mW
    "eth":       200,    # 有线网卡 ~200 mW (收发差别小)
}

# WiFi 典型吞吐量 (bytes/s)，用于估算传输时间
_NET_THROUGHPUT_BPS = {
    "wifi":  50 * 1024 * 1024,   # ~50 MB/s (802.11ac 实际)
    "lte":   10 * 1024 * 1024,   # ~10 MB/s (LTE Cat.6 实际)
    "eth":  100 * 1024 * 1024,   # ~100 MB/s (千兆以太网)
}

# 不同设备的典型 TDP (mW)，用于进程级能耗估算
_DEVICE_TDP_MW = {
    # Apple Silicon (CPU+GPU+ANE 整体封装功耗)
    "m1":        20_000,
    "m1_pro":    30_000,
    "m1_max":    40_000,
    "m1_ultra":  60_000,
    "m2":        20_000,
    "m2_pro":    30_000,
    "m2_max":    40_000,
    "m2_ultra":  60_000,
    "m3":        22_000,
    "m3_pro":    36_000,
    "m3_max":    44_000,
    "m3_ultra":  80_000,
    "m4":        22_000,
    "m4_pro":    36_000,
    "m4_max":    46_000,
    # 默认值
    "apple_default": 30_000,
    "x86_default":   65_000,
}

# ============ GPU 能耗分析模型常量 ============
#
# 将 GPU 总能耗拆分为 **计算能耗 (ALU/SM)** 和 **存储能耗 (HBM/显存搬运)**
#
# ---- 核心思路: Analytical Model (分析模型) ----
#
# 之前用 NVML 利用率做比例分配的方案有根本性缺陷:
#   - 利用率低 ≠ 功耗低 (GPU 有大量静态功耗)
#   - gpu_util 和 mem_util 不是同一维度，不能混在一起做比例
#   - 利用率与功耗之间不是线性关系 (DVFS、功耗门控等)
#
# 正确做法是从 **物理操作量** 出发:
#
#   1. 计算 Transformer 推理的 FLOPs 和 Memory Bytes:
#        FLOPs_per_token ≈ 2 × P          (P = 模型参数量, 每个参数做 1 次 MAC = 2 FLOPs)
#        Bytes_per_token ≈ P × b           (b = 每个参数的字节数, 每次 forward 搬运全部权重)
#
#   2. 用硬件规格的单位能耗常数计算:
#        E_compute = total_FLOPs × energy_per_FLOP  (pJ/FLOP)
#        E_memory  = total_Bytes × energy_per_Byte  (pJ/Byte)
#
#   3. 总能耗 = NVML 实测值 (最可靠)
#      用分析模型的 E_compute : E_memory 比例来拆分实测总能耗
#
# 这些 pJ/FLOP 和 pJ/Byte 常数来自:
#   [1] Horowitz, "1.1 Computing's Energy Problem", ISSCC 2014
#       - 经典能耗表: 不同工艺节点的运算和存储单位能耗
#   [2] NVIDIA GPU 白皮书 (V100/A100/H100)
#       - 峰值 FLOPS / TDP 可反推实际 pJ/FLOP
#   [3] Jouppi et al., "TPU v4: An Optically Reconfigurable Supercomputer", ISCA 2023
#       - HBM 能耗约 3.9 pJ/bit ≈ 31 pJ/Byte
#   [4] Leng et al., "GPUWattch: Enabling Energy Optimizations in GPGPUs", ISCA 2013

# 各 GPU 架构的硬件能耗常数
# 格式: { arch_prefix: (pJ_per_FLOP, pJ_per_byte) }
#   pJ_per_FLOP:  FP16/BF16 单次乘加运算的能耗 (picoJoules)
#   pJ_per_byte:  从 HBM 读取 1 字节的能耗 (picoJoules)
#
# 推导方法:
#   pJ_per_FLOP ≈ TDP_W / peak_TFLOPS / 1e12 × 1e12
#     例: V100 TDP=300W, FP16 peak=125 TFLOPS → 300/125e12 × 1e12 = 2.4 pJ
#     但实际 ALU 只占 TDP 的 ~60%, 所以有效 pJ/FLOP ≈ 300*0.6/125e12*1e12 ≈ 1.44 pJ
#     取保守值 ~0.4 pJ (考虑到 Tensor Core 效率远高于标量)
#
#   pJ_per_byte ≈ HBM 功耗 / 带宽
#     例: V100 HBM2 功耗 ~60W, 带宽 900 GB/s → 60/900e9 × 1e12 ≈ 67 pJ
#     但这包含了控制器开销, 纯数据搬运约 10-20 pJ/byte
#     参考 Horowitz 2014: DRAM access ~1.3-2.6 nJ/64bits ≈ 20-40 pJ/byte
#
# 注: 这些值是 "有效能耗" (effective energy), 包含了芯片内部的各种开销,
#     不是理论最小值。不同工作负载下会有波动, 但数量级是准确的。
_GPU_ENERGY_SPECS = {
    # Volta (V100, 12nm, HBM2)
    #   FP16 peak: 125 TFLOPS, TDP: 300W, HBM2 BW: 900 GB/s
    "V100":    (0.4, 20.0),
    "Tesla V": (0.4, 20.0),
    # Ampere (A100, 7nm, HBM2e)
    #   FP16 peak: 312 TFLOPS, TDP: 400W, HBM2e BW: 2039 GB/s
    "A100":    (0.3, 16.0),
    "A10":     (0.35, 18.0),
    "A30":     (0.32, 17.0),
    # Ada Lovelace (L40/L4/RTX 4090, 4nm, GDDR6X)
    #   GDDR6X 能耗比 HBM 高
    "L40":     (0.25, 25.0),
    "L4":      (0.28, 22.0),
    "RTX 40":  (0.25, 25.0),
    # Hopper (H100, 4nm, HBM3)
    #   FP16 peak: 989 TFLOPS, TDP: 700W, HBM3 BW: 3350 GB/s
    "H100":    (0.15, 12.0),
    "H200":    (0.14, 11.0),
    # Jetson / 嵌入式
    "Orin":    (0.5, 30.0),
    "Xavier":  (0.6, 35.0),
    # 默认 (保守估计, 偏向中端 GPU)
    "default": (0.35, 20.0),
}


def _get_gpu_energy_specs(gpu_name: str) -> tuple:
    """根据 GPU 名称匹配硬件能耗常数 (pJ_per_FLOP, pJ_per_byte)"""
    gpu_name_upper = gpu_name.upper()
    for prefix, specs in _GPU_ENERGY_SPECS.items():
        if prefix == "default":
            continue
        if prefix.upper() in gpu_name_upper:
            return specs
    return _GPU_ENERGY_SPECS["default"]


# ============ Zeus 集成 ============
#
# Zeus (https://ml.energy/zeus) 使用 NVML 的硬件能耗计数器
# (nvmlDeviceGetTotalEnergyConsumption) 来测量 GPU 能耗,
# 比周期性采样功率再积分更准确:
#   - 采样方式: E ≈ Σ(P_i × Δt), 受采样频率和时间对齐影响
#   - 计数器方式: E = counter_end - counter_start, 硬件级精度
#
# Zeus 还支持在 H100+ GPU 上直接测量 HBM 内存功率:
#   - NVML_FI_DEV_POWER_AVERAGE + NVML_POWER_SCOPE_MEMORY
#   - V100/A100 不支持此功能, 返回 "Not Supported"
#
# 集成策略:
#   - Zeus 可用 → 用 Zeus 测总能耗 (最准确)
#   - Zeus 不可用 → 退回 pynvml 采样
#   - H100+ 且 Zeus 可测内存功率 → 直接用硬件值拆分 compute/memory
#   - 其他 GPU → 用分析模型 (Analytical Model) 拆分

def _try_create_zeus_monitor(gpu_indices: list) -> tuple:
    """
    尝试创建 Zeus 能耗监控器。

    Args:
        gpu_indices: GPU 索引列表, e.g. [0] 或 [0, 1, 2, 3]

    Returns:
        (zeus_monitor, zeus_gpus) 或 (None, None)
        zeus_monitor: ZeusMonitor 实例
        zeus_gpus:    Zeus GPU 对象列表 (用于查询内存功率等)
    """
    try:
        from zeus.monitor.energy import ZeusMonitor
        from zeus.device.gpu import get_gpus

        monitor = ZeusMonitor(gpu_indices=gpu_indices)
        gpus_mgr = get_gpus()
        # 获取对应的 GPU 对象
        gpu_objs = [gpus_mgr._gpus[i] for i in gpu_indices]

        # 验证硬件能耗计数器可用
        if gpu_objs[0].supportsGetTotalEnergyConsumption():
            return monitor, gpu_objs
        else:
            print("[EnergyMonitor] Zeus: GPU 不支持硬件能耗计数器, 退回 pynvml 采样")
            return None, None

    except ImportError:
        return None, None
    except Exception as e:
        print(f"[EnergyMonitor] Zeus 初始化失败: {e}, 退回 pynvml 采样")
        return None, None


def _check_zeus_memory_power(gpu_obj) -> bool:
    """
    检查 Zeus GPU 对象是否支持直接测量 HBM 内存功率。
    仅 H100+ GPU 支持。

    Returns:
        True if memory power measurement is supported.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mem_power = gpu_obj.getAverageMemoryPowerUsage()
            return mem_power > 0
    except Exception:
        return False


def _count_model_params(model) -> tuple:
    """
    统计模型参数量和每参数字节数。
    返回: (num_params, bytes_per_param)
    """
    if model is None:
        return 0, 0
    total_params = 0
    total_bytes = 0
    try:
        for p in model.parameters():
            total_params += p.numel()
            total_bytes += p.numel() * p.element_size()
    except Exception:
        return 0, 0
    bytes_per_param = total_bytes / total_params if total_params > 0 else 2
    return total_params, bytes_per_param


def _extract_model_arch(model) -> dict:
    """
    从 model.config 提取 Transformer 结构参数, 用于精细化 memory bytes 计算。

    返回 dict 包含:
      - hidden_size (d_model)
      - intermediate_size (FFN 中间层维度)
      - num_hidden_layers (L)
      - num_attention_heads
      - num_key_value_heads (GQA/MQA, 若无则 = num_attention_heads)
      - head_dim
      - vocab_size

    如果无法获取, 返回空 dict, 调用方退回到粗糙估算。

    支持多种命名约定:
      - Qwen/LLaMA/Mistral: hidden_size, num_attention_heads, ...
      - GPT-2/GPT-Neo:       n_embd, n_head, n_layer, n_inner
    """
    arch = {}
    try:
        cfg = model.config
    except AttributeError:
        return arch

    # 带别名的必需字段: (标准名, 备选名列表)
    _FIELD_ALIASES = {
        "hidden_size":       ["n_embd", "d_model"],
        "intermediate_size": ["n_inner", "d_inner", "ffn_dim"],
        "num_hidden_layers": ["n_layer", "num_layers", "n_layers"],
        "num_attention_heads": ["n_head", "num_heads", "n_heads"],
        "vocab_size":        [],  # 几乎所有模型都用 vocab_size
    }

    for key, aliases in _FIELD_ALIASES.items():
        val = getattr(cfg, key, None)
        if val is None:
            # 尝试备选名
            for alias in aliases:
                val = getattr(cfg, alias, None)
                if val is not None:
                    break
        if val is None:
            # intermediate_size 可以从 hidden_size 推导 (默认 4x)
            if key == "intermediate_size" and "hidden_size" in arch:
                val = arch["hidden_size"] * 4
            else:
                return {}  # 缺少关键字段, 放弃精细化
        arch[key] = int(val)

    # GQA: num_key_value_heads, 默认 = num_attention_heads (MHA)
    arch["num_key_value_heads"] = int(
        getattr(cfg, "num_key_value_heads", None) or arch["num_attention_heads"]
    )

    # head_dim: 优先从 config 读, 否则 = hidden_size / num_attention_heads
    hd = getattr(cfg, "head_dim", None)
    if hd is not None:
        arch["head_dim"] = int(hd)
    else:
        arch["head_dim"] = arch["hidden_size"] // arch["num_attention_heads"]

    return arch


# ============ 芯片检测 ============

def _detect_apple_chip() -> str:
    """检测 Apple Silicon 芯片型号，返回 TDP 表中的 key"""
    if platform.system() != "Darwin":
        return ""
    try:
        r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                           capture_output=True, text=True, timeout=3)
        brand = r.stdout.strip().lower()  # e.g. "Apple M3 Max"
        # 解析: "apple m3 max" → "m3_max"
        for token in brand.split():
            if token.startswith("m") and token[1:].isdigit():
                gen = token  # e.g. "m3"
                # 看后面有没有 pro/max/ultra
                parts = brand.split(token, 1)
                if len(parts) > 1:
                    suffix = parts[1].strip().split()[0] if parts[1].strip() else ""
                    if suffix in ("pro", "max", "ultra"):
                        return f"{gen}_{suffix}"
                return gen
    except Exception:
        pass
    return ""


# ============ EnergyMonitor ============

class EnergyMonitor:
    """
    本地推理能耗监控器（UAV 端）—— 进程级隔离

    核心原理:
      所有指标都是 **进程级** 的，不受其他应用影响:
      - CPU 时间: resource.getrusage(RUSAGE_SELF) — 仅统计本进程
      - GPU 显存: torch.cuda 的 per-process 统计
      - 能耗估算: 进程 CPU 时间 × 设备 TDP 占比（而非全系统功率）

    可选的硬件级参考数据 (会被其他应用影响，仅作参考):
      - macOS powermetrics: 全芯片功率 (标注为 sys_*)
      - NVIDIA pynvml:      全 GPU 卡功率 (标注为 sys_*)

    用法:
        mon = EnergyMonitor(device=torch.device("mps"), interval=0.5)
        mon.start()
        # ... 推理代码 ...
        stats = mon.stop()
        print(EnergyMonitor.format_report(stats))
    """

    def __init__(self, device: Optional[torch.device] = None,
                 framework: str = "pytorch", interval: float = 0.5,
                 model=None, gpu_indices: list = None):
        """
        Args:
            device:      推理设备 (torch.device)
            framework:   "pytorch" / "mlx" / "vllm"
            interval:    NVML 采样间隔 (秒), 仅在 Zeus 不可用时使用
            model:       推理模型 (torch.nn.Module), 用于统计参数量和 dtype,
                         以便用分析模型计算 FLOPs/Bytes 能耗拆分
            gpu_indices: GPU 索引列表, e.g. [0] 或 [0, 1, 2, 3]
                         如果不传, 自动从 device 推断
        """
        self.device = device
        self.framework = framework
        self.interval = interval

        # 判断是否为 GPU 密集型任务
        self._is_gpu_intensive = (
            framework in ("mlx", "vllm")
            or (device is not None and device.type in ("mps", "cuda"))
        )

        # 状态
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._power_samples: list = []
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._rusage_start = None
        self._rusage_end = None
        self._tokens_generated: int = 0  # 生成的 token 数 (由 stop() 传入)

        # CUDA 相关
        self._nvml_handle = None
        self._gpu_name: str = ""
        self._gpu_energy_specs: tuple = _GPU_ENERGY_SPECS["default"]

        # ---- Zeus 集成 ----
        # 优先使用 Zeus 的硬件能耗计数器 (比 pynvml 采样更准确)
        self._zeus_monitor = None       # ZeusMonitor 实例
        self._zeus_gpus = None          # Zeus GPU 对象列表
        self._zeus_mem_power_ok = False  # 是否支持直接测量 HBM 内存功率 (H100+)
        self._zeus_window_name = None   # 当前 Zeus 测量窗口名称

        # 推断 GPU 索引
        if gpu_indices is None:
            if device is not None and device.type == "cuda":
                gpu_indices = [device.index or 0]
            elif framework == "vllm" and torch.cuda.is_available():
                gpu_indices = list(range(torch.cuda.device_count()))
        self._gpu_indices = gpu_indices or []

        # 尝试初始化 Zeus
        if self._is_gpu_intensive and self._gpu_indices:
            self._zeus_monitor, self._zeus_gpus = _try_create_zeus_monitor(
                self._gpu_indices)
            if self._zeus_monitor is not None:
                self._zeus_mem_power_ok = _check_zeus_memory_power(
                    self._zeus_gpus[0])
                backend_info = "Zeus (硬件能耗计数器)"
                if self._zeus_mem_power_ok:
                    backend_info += " + HBM 内存功率直测 (H100+)"
                else:
                    backend_info += " + 分析模型拆分 (V100/A100)"
                print(f"[EnergyMonitor] 能耗后端: {backend_info}")
                print(f"[EnergyMonitor] 监控 GPU: {self._gpu_indices}")

        # 模型参数统计 (用于分析模型)
        self._num_params, self._bytes_per_param = _count_model_params(model)
        self._model_arch: dict = _extract_model_arch(model)
        if self._num_params > 0:
            arch_info = ""
            if self._model_arch:
                a = self._model_arch
                arch_info = (f", arch: L={a['num_hidden_layers']} "
                             f"d={a['hidden_size']} "
                             f"kv_heads={a['num_key_value_heads']} "
                             f"head_dim={a['head_dim']}")
            print(f"[EnergyMonitor] Model: {self._num_params/1e9:.2f}B params, "
                  f"{self._bytes_per_param:.1f} bytes/param "
                  f"({self._bytes_per_param*8:.0f}-bit){arch_info}")

        # 空闲功率基线 (用于分离静态/动态功耗)
        self._idle_power_mw: float = 0.0

        # 设备 TDP
        self._device_tdp_mw = self._get_device_tdp()

        # 硬件级功率后端 (可选参考, Zeus 不可用时作为主后端)
        self._hw_backend = self._detect_hw_backend()

        # 获取 GPU 硬件能耗常数 (CUDA 设备)
        if self.device is not None and self.device.type == "cuda":
            try:
                idx = self.device.index or 0
                self._gpu_name = torch.cuda.get_device_name(idx)
                self._gpu_energy_specs = _get_gpu_energy_specs(self._gpu_name)
            except Exception:
                pass
        elif framework == "vllm" and torch.cuda.is_available():
            try:
                self._gpu_name = torch.cuda.get_device_name(0)
                self._gpu_energy_specs = _get_gpu_energy_specs(self._gpu_name)
            except Exception:
                pass

    # ---------- TDP 检测 ----------

    def _get_device_tdp(self) -> float:
        """获取当前设备的 TDP (mW)"""
        if platform.system() == "Darwin":
            chip = _detect_apple_chip()
            if chip and chip in _DEVICE_TDP_MW:
                return _DEVICE_TDP_MW[chip]
            return _DEVICE_TDP_MW["apple_default"]

        if self.device is not None and self.device.type == "cuda":
            try:
                import pynvml
                pynvml.nvmlInit()
                idx = self.device.index or 0
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                tdp = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                pynvml.nvmlShutdown()
                return tdp
            except Exception:
                pass

        return _DEVICE_TDP_MW["x86_default"]

    # ---------- 硬件级后端探测 ----------

    def _detect_hw_backend(self) -> str:
        """检测可用的硬件级功率监控后端（可选参考，非进程级）"""
        if self.device is not None and self.device.type == "cuda":
            try:
                import pynvml
                pynvml.nvmlInit()
                idx = self.device.index or 0
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                return "nvml"
            except Exception:
                pass

        if platform.system() == "Darwin":
            try:
                r = subprocess.run(
                    ["sudo", "-n", "powermetrics", "-n", "1", "-i", "100",
                     "--samplers", "cpu_power"],
                    capture_output=True, timeout=5)
                if r.returncode == 0:
                    return "powermetrics"
            except Exception:
                pass

        return "none"

    # ---------- 硬件级采样线程 ----------

    def _sample_nvml(self):
        """
        NVML 采样: 返回 (cpu_mw, total_gpu_mw, gpu_util%, mem_util%)
        gpu_util: SM 核心利用率 (0-100) → 反映 ALU 计算负载
        mem_util: 显存带宽利用率 (0-100) → 反映数据搬运负载
        """
        import pynvml
        power_mw = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)  # 单位: mW
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            gpu_util = util.gpu   # SM utilization %
            mem_util = util.memory  # Memory BW utilization %
        except Exception:
            gpu_util = 0
            mem_util = 0
        return 0, power_mw, gpu_util, mem_util

    def _sample_powermetrics(self):
        """macOS powermetrics 采样 (不支持 compute/memory 拆分)"""
        try:
            r = subprocess.run(
                ["sudo", "-n", "powermetrics", "-n", "1", "-i", "100",
                 "--samplers", "cpu_power"],
                capture_output=True, text=True, timeout=5)
            cpu_mw = gpu_mw = ane_mw = 0
            for line in r.stdout.splitlines():
                low = line.strip().lower()
                if "cpu power:" in low and "mw" in low:
                    cpu_mw = self._parse_mw(line)
                elif "gpu power:" in low and "mw" in low:
                    gpu_mw = self._parse_mw(line)
                elif "ane power:" in low and "mw" in low:
                    ane_mw = self._parse_mw(line)
            return cpu_mw, gpu_mw + ane_mw, 0, 0
        except Exception:
            return 0, 0, 0, 0

    @staticmethod
    def _parse_mw(line: str) -> float:
        try:
            parts = line.split(":")
            if len(parts) >= 2:
                num_str = parts[1].strip().split()[0]
                return float(num_str)
        except (ValueError, IndexError):
            pass
        return 0.0

    def _sampling_loop(self):
        sample_fn = {
            "nvml": self._sample_nvml,
            "powermetrics": self._sample_powermetrics,
        }.get(self._hw_backend)

        if sample_fn is None:
            return

        while self._running:
            try:
                cpu_mw, gpu_mw, gpu_util, mem_util = sample_fn()
                self._power_samples.append(
                    (time.time(), cpu_mw, gpu_mw, gpu_util, mem_util))
            except Exception:
                pass
            time.sleep(self.interval)

    # ---------- 公共 API ----------

    def _measure_idle_power(self, n_samples: int = 5, interval: float = 0.1) -> float:
        """
        在推理开始前测量 GPU 空闲功率基线 (mW)。

        原理: GPU 即使空闲也有静态功耗 (漏电流、时钟树、PHY 等),
        典型值约为 TDP 的 10-20%。通过实测空闲功率, 可以把总能耗拆分为:
          - 静态能耗 (idle): 不管算不算都会消耗的
          - 动态能耗 (active): 只有计算/搬运时才产生的

        只有动态能耗才应该按 compute:memory 比例拆分。
        """
        if self._hw_backend != "nvml" or self._nvml_handle is None:
            return 0.0

        import pynvml
        readings = []
        for _ in range(n_samples):
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)  # mW
                readings.append(power)
            except Exception:
                pass
            time.sleep(interval)

        if readings:
            # 取中位数, 比均值更抗噪
            readings.sort()
            mid = len(readings) // 2
            return readings[mid]
        return 0.0

    def start(self):
        """开始能耗监控"""
        self._power_samples = []
        self._start_time = time.time()
        self._rusage_start = resource.getrusage(resource.RUSAGE_SELF)

        # 在推理开始前测量空闲功率基线 (方案 B)
        if self._hw_backend == "nvml":
            self._idle_power_mw = self._measure_idle_power()
            if self._idle_power_mw > 0:
                print(f"[EnergyMonitor] Idle power baseline: "
                      f"{self._idle_power_mw:.0f} mW")

        # ---- Zeus: 开始测量窗口 ----
        if self._zeus_monitor is not None:
            self._zeus_window_name = f"inference_{id(self)}_{time.monotonic_ns()}"
            self._zeus_monitor.begin_window(self._zeus_window_name)

        # ---- pynvml 采样线程 (Zeus 不可用时, 或作为辅助参考) ----
        if self._hw_backend != "none":
            self._running = True
            self._thread = threading.Thread(target=self._sampling_loop, daemon=True)
            self._thread.start()

        if self.device is not None and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def stop(self, tokens_generated: int = 0,
             avg_seq_len: int = 0) -> Dict[str, float]:
        """
        停止监控并返回能耗统计字典。

        使用 **精细化分析模型 (Refined Analytical Model)** 拆分能耗:

        方案 A — 精细化 Memory Bytes 计算:
          将每 token 的 HBM 搬运量拆分为 4 个组成部分:
            ① 权重搬运:  P × b               (每次 forward 搬全部权重)
            ② KV cache 读: 2 × L × d_kv × seq_len × b  (attention 读历史)
            ③ KV cache 写: 2 × L × d_kv × 1 × b        (写新 token 的 K,V)
            ④ 激活值搬运:  2L(d_model + d_inter) × b    (hidden + FFN)
          其中 d_kv = num_kv_heads × head_dim

        方案 B — 空闲功率基线:
          将 NVML 实测总能耗拆分为:
            - 静态能耗 (idle):  idle_power × wall_time  (不管算不算都消耗)
            - 动态能耗 (active): total - idle            (计算/搬运产生的)
          只有动态能耗才按 compute:memory 理论比例拆分。

        Args:
            tokens_generated: 本次推理生成的 token 数量
            avg_seq_len:      生成过程中的平均上下文长度 (用于 KV cache 估算)
                              若为 0, 则自动估算为 tokens_generated / 2

        返回 dict 包含:
          - compute_energy_mj:  ALU/SM 计算能耗
          - memory_energy_mj:   HBM/显存搬运能耗 (权重 + KV cache + 激活值)
          - idle_energy_mj:     静态/不可归因能耗
          - est_energy_mj:      总能耗 = compute + memory + idle
        """
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None

        self._end_time = time.time()
        self._rusage_end = resource.getrusage(resource.RUSAGE_SELF)

        wall_time = self._end_time - self._start_time
        self._tokens_generated = tokens_generated

        # 进程级 CPU 时间
        u0 = self._rusage_start
        u1 = self._rusage_end
        cpu_user = u1.ru_utime - u0.ru_utime
        cpu_sys = u1.ru_stime - u0.ru_stime
        cpu_total = cpu_user + cpu_sys
        cpu_util = (cpu_total / wall_time * 100) if wall_time > 0 else 0

        # 峰值内存
        peak_rss = u1.ru_maxrss
        if platform.system() == "Darwin":
            peak_mem_mb = peak_rss / (1024 * 1024)
        else:
            peak_mem_mb = peak_rss / 1024

        # ========== 方案 A: 精细化分析模型 ==========
        #
        # Decoder-only Transformer 自回归推理 (每生成 1 个 token):
        #
        # ---- 计算量 (FLOPs) ----
        #   FLOPs_per_token ≈ 2 × P   (P = 参数量, 每参数 1 MAC = 2 FLOPs)
        #
        # ---- 显存搬运量 (Bytes) ----
        #   ① 权重搬运:     P × b                                (搬全部权重)
        #   ② KV cache 读:  2 × L × d_kv × seq_len × b          (读历史 K,V)
        #   ③ KV cache 写:  2 × L × d_kv × 1 × b                (写新 K,V)
        #   ④ 激活值搬运:   2 × L × (d_model + d_inter) × b     (hidden+FFN)
        #
        #   其中:
        #     L = num_hidden_layers
        #     d_kv = num_key_value_heads × head_dim  (GQA 下远小于 d_model)
        #     d_model = hidden_size
        #     d_inter = intermediate_size (FFN 中间层)
        #     b = bytes_per_param
        #     seq_len = 平均上下文长度 (随生成逐渐增长)
        #
        # 如果无法获取 model.config, 退回到粗糙估算 (只算权重搬运)。

        pj_per_flop, pj_per_byte = self._gpu_energy_specs
        num_params = self._num_params
        bytes_per_param = self._bytes_per_param
        arch = self._model_arch

        # 平均 seq_len: 如果调用方没传, 用 tokens_generated/2 做粗略估算
        # (decode 阶段 seq_len 从 prompt_len 增长到 prompt_len + tokens_generated,
        #  平均值约为 prompt_len + tokens_generated/2, 这里忽略 prompt_len)
        if avg_seq_len <= 0 and tokens_generated > 0:
            avg_seq_len = max(tokens_generated // 2, 1)

        # ---- 计算每 token 的 FLOPs ----
        flops_per_token = 2 * num_params if num_params > 0 else 0

        # ---- 计算每 token 的 Bytes (精细化) ----
        weight_bytes_per_tok = 0     # ① 权重读取 (每token前向传播需从HBM读全部权重到SM)
        kv_read_bytes_per_tok = 0    # ② KV cache 读
        kv_write_bytes_per_tok = 0   # ③ KV cache 写
        activation_bytes_per_tok = 0  # ④ 激活值搬运

        if num_params > 0:
            b = bytes_per_param
            # ① 权重读取: decode 阶段每生成1个token，GPU都需要从HBM读取全部权重到SM做矩阵-向量乘法
            #    (不是模型加载，而是推理时的数据搬运；权重远超L2 cache，无法复用)
            weight_bytes_per_tok = num_params * b

            if arch:
                # 精细化: 从 model.config 获取结构参数
                L = arch["num_hidden_layers"]
                d_model = arch["hidden_size"]
                d_inter = arch["intermediate_size"]
                n_kv_heads = arch["num_key_value_heads"]
                head_dim = arch["head_dim"]
                d_kv = n_kv_heads * head_dim

                kv_read_bytes_per_tok = 2 * L * d_kv * avg_seq_len * b   # ②
                kv_write_bytes_per_tok = 2 * L * d_kv * 1 * b            # ③
                activation_bytes_per_tok = 2 * L * (d_model + d_inter) * b  # ④

        total_bytes_per_tok = (weight_bytes_per_tok + kv_read_bytes_per_tok
                               + kv_write_bytes_per_tok + activation_bytes_per_tok)

        # 总操作量
        total_flops = flops_per_token * tokens_generated
        total_bytes = total_bytes_per_tok * tokens_generated

        # 理论能耗 (pJ → mJ: ÷ 1e9)
        theory_compute_mj = total_flops * pj_per_flop / 1e9
        theory_memory_mj = total_bytes * pj_per_byte / 1e9
        theory_total_mj = theory_compute_mj + theory_memory_mj

        # 计算 compute : memory 的理论比例
        if theory_total_mj > 0:
            compute_ratio = theory_compute_mj / theory_total_mj
            memory_ratio = theory_memory_mj / theory_total_mj
        else:
            compute_ratio = 0.3
            memory_ratio = 0.7

        # ========== 实测总能耗 + 能耗拆分 ==========
        #
        # 优先级:
        #   1. Zeus 硬件能耗计数器 (nvmlDeviceGetTotalEnergyConsumption)
        #      → 最准确, 基于 GPU 内部硬件计数器, 不受采样频率影响
        #   2. pynvml 周期性采样 + 积分
        #      → Zeus 不可用时的退化方案
        #   3. TDP × wall_time × load_factor
        #      → 最后退化方案 (无 NVML)
        #
        # 能耗拆分:
        #   - H100+ 且 Zeus 可测内存功率 → 硬件直接拆分
        #   - 其他 GPU → 分析模型 (Analytical Model) 拆分动态能耗

        compute_energy_mj = 0.0
        memory_energy_mj = 0.0
        idle_energy_mj = 0.0
        est_energy_mj = 0.0
        avg_gpu_util = 0.0
        avg_mem_util = 0.0
        est_method = "unknown"

        # 收集 pynvml 采样数据 (不管 Zeus 是否可用, 都收集作为参考)
        n_samples = len(self._power_samples)
        if n_samples > 0:
            gpu_utils = [s[3] for s in self._power_samples]
            mem_utils = [s[4] for s in self._power_samples]
            avg_gpu_util = sum(gpu_utils) / len(gpu_utils)
            avg_mem_util = sum(mem_utils) / len(mem_utils)

        # ---- 辅助函数: 用分析模型拆分动态能耗 ----
        def _split_dynamic(total_mj: float, method_prefix: str) -> tuple:
            """
            将总能耗拆分为 (compute, memory, idle, method_str)。
            用空闲功率基线分离静态能耗, 再用分析模型比例拆分动态部分。
            """
            _idle = self._idle_power_mw * wall_time  # mW × s = mJ
            # 防止 idle 超过总量 (采样噪声)
            if _idle >= total_mj * 0.95:
                _idle = total_mj * 0.15  # 保守回退
            _dynamic = total_mj - _idle
            _compute = _dynamic * compute_ratio
            _memory = _dynamic * memory_ratio
            _split = "refined" if arch else "coarse"
            _method = (
                f"{method_prefix}_analytical_{_split} "
                f"(idle={self._idle_power_mw:.0f}mW, "
                f"dynamic={_dynamic:.0f}mJ, "
                f"ratio={compute_ratio:.1%}:{memory_ratio:.1%})"
            )
            if arch:
                _method += (
                    f" [KV: L={arch['num_hidden_layers']} "
                    f"d_kv={arch['num_key_value_heads']*arch['head_dim']} "
                    f"avg_seq={avg_seq_len}]"
                )
            return _compute, _memory, _idle, _method

        if self._is_gpu_intensive and self.device and self.device.type == "cuda":

            # =============================================
            # 优先级 1: Zeus 硬件能耗计数器
            # =============================================
            zeus_ok = False
            if self._zeus_monitor is not None and self._zeus_window_name is not None:
                try:
                    zeus_result = self._zeus_monitor.end_window(
                        self._zeus_window_name, sync_execution=True)
                    gpu_idx = self._gpu_indices[0] if self._gpu_indices else 0
                    zeus_gpu_energy_j = zeus_result.gpu_energy.get(gpu_idx, 0.0)

                    # 多卡: 累加所有监控 GPU 的能耗
                    if len(self._gpu_indices) > 1:
                        zeus_gpu_energy_j = sum(
                            zeus_result.gpu_energy.get(i, 0.0)
                            for i in self._gpu_indices
                        )

                    est_energy_mj = zeus_gpu_energy_j * 1000.0  # J → mJ
                    zeus_ok = True

                    print(f"[EnergyMonitor] Zeus 硬件计数器: "
                          f"GPU 总能耗 = {zeus_gpu_energy_j:.4f} J "
                          f"({est_energy_mj:.1f} mJ)")

                    # ---- H100+: 尝试硬件直接拆分 compute/memory ----
                    if self._zeus_mem_power_ok and self._zeus_gpus:
                        try:
                            # 获取 HBM 内存平均功率 (mW)
                            mem_power_mw = self._zeus_gpus[0].getAverageMemoryPowerUsage()
                            if mem_power_mw > 0:
                                # 硬件测量的内存能耗
                                hw_mem_energy_mj = mem_power_mw * wall_time  # mW × s = mJ
                                # 确保不超过总能耗的 90%
                                if hw_mem_energy_mj > est_energy_mj * 0.90:
                                    hw_mem_energy_mj = est_energy_mj * 0.50
                                memory_energy_mj = hw_mem_energy_mj
                                # 空闲功率
                                idle_energy_mj = self._idle_power_mw * wall_time
                                if idle_energy_mj >= est_energy_mj * 0.95:
                                    idle_energy_mj = est_energy_mj * 0.15
                                # 剩余为计算能耗
                                compute_energy_mj = max(
                                    0.0, est_energy_mj - memory_energy_mj - idle_energy_mj)
                                est_method = (
                                    f"zeus_hw_total + hw_mem_split "
                                    f"(mem_power={mem_power_mw:.0f}mW, "
                                    f"idle={self._idle_power_mw:.0f}mW)")
                                print(f"[EnergyMonitor] H100+ 硬件内存功率: "
                                      f"{mem_power_mw:.0f} mW → "
                                      f"memory={memory_energy_mj:.0f} mJ")
                            else:
                                raise ValueError("mem_power_mw == 0")
                        except Exception as e:
                            # H100+ 内存功率测量失败, 退回分析模型拆分
                            print(f"[EnergyMonitor] H100+ 内存功率测量失败: {e}, "
                                  f"退回分析模型拆分")
                            if num_params > 0 and tokens_generated > 0:
                                compute_energy_mj, memory_energy_mj, \
                                    idle_energy_mj, est_method = \
                                    _split_dynamic(est_energy_mj, "zeus_hw_total")
                            else:
                                est_method = "zeus_hw_total (no model info for split)"
                    else:
                        # ---- 非 H100+: 用分析模型拆分 ----
                        if num_params > 0 and tokens_generated > 0:
                            compute_energy_mj, memory_energy_mj, \
                                idle_energy_mj, est_method = \
                                _split_dynamic(est_energy_mj, "zeus_hw_total")
                        else:
                            est_method = "zeus_hw_total (no model info for split)"

                except Exception as e:
                    print(f"[EnergyMonitor] Zeus 测量失败: {e}, 退回 pynvml 采样")
                    self._zeus_window_name = None  # 防止重复 end_window

            # =============================================
            # 优先级 2: pynvml 周期性采样 + 积分
            # =============================================
            if not zeus_ok and self._hw_backend == "nvml" and n_samples > 0:
                dt = wall_time / n_samples if n_samples > 0 else 0
                total_energy_mj = 0.0
                for sample_data in self._power_samples:
                    _, _cpu_mw, gpu_mw, _g, _m = sample_data
                    total_energy_mj += gpu_mw * dt  # mW × s = mJ
                est_energy_mj = total_energy_mj

                if num_params > 0 and tokens_generated > 0:
                    compute_energy_mj, memory_energy_mj, \
                        idle_energy_mj, est_method = \
                        _split_dynamic(est_energy_mj, "nvml_sampling")
                    est_method += f" (samples={n_samples})"
                else:
                    est_method = (f"nvml_total_only "
                                  f"({n_samples} samples, no model info for split)")

            # =============================================
            # 优先级 3: TDP 估算
            # =============================================
            elif not zeus_ok and est_energy_mj <= 0:
                gpu_load_factor = 0.85
                est_energy_mj = wall_time * self._device_tdp_mw * gpu_load_factor
                if num_params > 0 and tokens_generated > 0:
                    compute_energy_mj = est_energy_mj * compute_ratio
                    memory_energy_mj = est_energy_mj * memory_ratio
                    idle_energy_mj = 0.0
                    est_method = (f"tdp_analytical "
                                  f"(wall×TDP×{gpu_load_factor}, "
                                  f"ratio={compute_ratio:.1%}:{memory_ratio:.1%})")
                else:
                    compute_energy_mj = est_energy_mj
                    est_method = f"gpu_tdp (wall×TDP×{gpu_load_factor})"

        elif self._is_gpu_intensive:
            # ---- 非 CUDA GPU (e.g., MLX on Apple Silicon) ----
            if platform.system() == "Darwin":
                gpu_load_factor = 0.70
            else:
                gpu_load_factor = 0.85
            est_energy_mj = wall_time * self._device_tdp_mw * gpu_load_factor

            if num_params > 0 and tokens_generated > 0:
                compute_energy_mj = est_energy_mj * compute_ratio
                memory_energy_mj = est_energy_mj * memory_ratio
                idle_energy_mj = 0.0
                est_method = (f"tdp_analytical "
                              f"(wall×TDP×{gpu_load_factor}, "
                              f"ratio={compute_ratio:.1%}:{memory_ratio:.1%})")
            else:
                compute_energy_mj = est_energy_mj
                memory_energy_mj = 0.0
                idle_energy_mj = 0.0
                est_method = f"gpu_tdp (wall_time × TDP × {gpu_load_factor})"
        else:
            # ---- CPU 密集型 ----
            est_energy_mj = cpu_total * self._device_tdp_mw
            compute_energy_mj = est_energy_mj
            memory_energy_mj = 0.0
            idle_energy_mj = 0.0
            est_method = "cpu (cpu_time × TDP)"

        est_avg_power_mw = est_energy_mj / wall_time if wall_time > 0 else 0

        stats: Dict[str, float] = {
            "wall_time": round(wall_time, 3),
            "cpu_user_time": round(cpu_user, 3),
            "cpu_sys_time": round(cpu_sys, 3),
            "cpu_total_time": round(cpu_total, 3),
            "cpu_utilization": round(cpu_util, 1),
            "peak_memory_mb": round(peak_mem_mb, 1),
            "device_tdp_mw": round(self._device_tdp_mw, 0),
            "gpu_intensive": self._is_gpu_intensive,
            "est_method": est_method,
            # 总能耗
            "est_energy_mj": round(est_energy_mj, 1),
            "est_avg_power_mw": round(est_avg_power_mw, 1),
            # 计算 vs 存储 vs 静态 拆分
            "compute_energy_mj": round(compute_energy_mj, 1),
            "memory_energy_mj": round(memory_energy_mj, 1),
            "idle_energy_mj": round(idle_energy_mj, 1),
        }

        # 分析模型详情
        if num_params > 0 and tokens_generated > 0:
            stats.update({
                "model_params_B": round(num_params / 1e9, 3),
                "model_bytes_per_param": round(bytes_per_param, 1),
                "flops_per_token": flops_per_token,
                "pj_per_flop": pj_per_flop,
                "pj_per_byte": pj_per_byte,
                "total_flops": total_flops,
                "total_memory_bytes": total_bytes,
                "theory_compute_mj": round(theory_compute_mj, 3),
                "theory_memory_mj": round(theory_memory_mj, 3),
                "compute_ratio": round(compute_ratio, 4),
                "memory_ratio": round(memory_ratio, 4),
                "avg_seq_len": avg_seq_len,
            })
            # 精细化 memory breakdown
            if arch:
                stats.update({
                    "weight_bytes_per_tok": weight_bytes_per_tok,
                    "kv_read_bytes_per_tok": kv_read_bytes_per_tok,
                    "kv_write_bytes_per_tok": kv_write_bytes_per_tok,
                    "activation_bytes_per_tok": activation_bytes_per_tok,
                    "total_bytes_per_tok": total_bytes_per_tok,
                })

        # GPU 利用率统计 (仍然采集, 作为参考)
        if avg_gpu_util > 0 or avg_mem_util > 0:
            stats["avg_gpu_util"] = round(avg_gpu_util, 1)
            stats["avg_mem_util"] = round(avg_mem_util, 1)

        # GPU 信息
        if self._gpu_name:
            stats["gpu_name"] = self._gpu_name
            stats["gpu_energy_specs"] = (f"pJ/FLOP={pj_per_flop}, "
                                          f"pJ/Byte={pj_per_byte}")

        # 空闲功率基线
        if self._idle_power_mw > 0:
            stats["idle_power_mw"] = round(self._idle_power_mw, 0)

        # CUDA 峰值显存
        if self.device is not None and self.device.type == "cuda":
            peak_gpu_mem = torch.cuda.max_memory_allocated(self.device) / (1024**2)
            stats["peak_gpu_memory_mb"] = round(peak_gpu_mem, 1)

        # 硬件级参考数据 (原始采样)
        stats["sys_hw_backend"] = self._hw_backend

        if self._power_samples:
            gpu_vals = [s[2] for s in self._power_samples]
            cpu_vals = [s[1] for s in self._power_samples]
            total_vals = [c + g for c, g in zip(cpu_vals, gpu_vals)]

            avg_total = sum(total_vals) / len(total_vals)
            peak_total = max(total_vals)
            sys_energy_mj = avg_total * wall_time

            stats.update({
                "sys_avg_power_mw": round(avg_total, 1),
                "sys_peak_power_mw": round(peak_total, 1),
                "sys_energy_mj": round(sys_energy_mj, 1),
                "sys_power_samples": len(self._power_samples),
            })

        # 清理 nvml
        if self._hw_backend == "nvml":
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass

        return stats

    # ---------- 网络能耗估算 ----------

    @staticmethod
    def estimate_network_energy(tx_bytes: int, rx_bytes: int,
                                net_type: str = "wifi") -> Dict[str, float]:
        """
        估算网络传输能耗 (进程级 — 只算本进程实际传输的字节数)

        Args:
            tx_bytes: 上行字节数 (UAV → BS)
            rx_bytes: 下行字节数 (BS → UAV)
            net_type: 网络类型 "wifi" / "lte" / "eth"
        """
        throughput = _NET_THROUGHPUT_BPS.get(net_type, _NET_THROUGHPUT_BPS["wifi"])

        if net_type == "eth":
            tx_power = rx_power = _NET_POWER_MW["eth"]
        elif net_type == "lte":
            tx_power = _NET_POWER_MW["lte_tx"]
            rx_power = _NET_POWER_MW["lte_rx"]
        else:
            tx_power = _NET_POWER_MW["wifi_tx"]
            rx_power = _NET_POWER_MW["wifi_rx"]

        tx_time = tx_bytes / throughput if throughput > 0 else 0
        rx_time = rx_bytes / throughput if throughput > 0 else 0

        tx_energy = tx_power * tx_time
        rx_energy = rx_power * rx_time

        return {
            "net_tx_bytes": tx_bytes,
            "net_rx_bytes": rx_bytes,
            "net_total_bytes": tx_bytes + rx_bytes,
            "net_tx_energy_mj": round(tx_energy, 2),
            "net_rx_energy_mj": round(rx_energy, 2),
            "net_total_energy_mj": round(tx_energy + rx_energy, 2),
            "net_type": net_type,
        }

    # ---------- 报告格式化 ----------

    @staticmethod
    def format_report(stats: Dict[str, float], tokens_generated: int = 0,
                      net_stats: Optional[Dict] = None) -> str:
        """
        格式化输出能耗报告。

        包含:
          - 进程级指标 (CPU, 内存)
          - 精细化分析模型 (FLOPs, Memory Bytes 四部分拆分)
          - 能耗三部分: compute (ALU) + memory (HBM) + idle (静态)
          - 网络传输能耗
          - 汇总
        """
        is_gpu = stats.get("gpu_intensive", False)
        est_method = stats.get("est_method", "unknown")

        lines = [
            "",
            "  [进程级指标 — 仅本进程, 不受其他应用影响]",
            f"  ⏱  Wall time:          {stats['wall_time']:.2f}s",
            f"  🖥  CPU time (process): {stats['cpu_total_time']:.2f}s "
            f"(user {stats['cpu_user_time']:.2f}s + sys {stats['cpu_sys_time']:.2f}s)",
            f"  📈 CPU utilization:     {stats['cpu_utilization']:.1f}%",
            f"  💾 Peak memory (RSS):   {stats['peak_memory_mb']:.1f} MB",
        ]

        if "peak_gpu_memory_mb" in stats:
            lines.append(
                f"  🎮 Peak GPU memory:     {stats['peak_gpu_memory_mb']:.1f} MB")

        # GPU 利用率 (参考)
        if "avg_gpu_util" in stats:
            lines.append(
                f"  📊 Avg GPU SM util:     {stats['avg_gpu_util']:.1f}% (参考)")
            lines.append(
                f"  📊 Avg GPU Mem BW util: {stats['avg_mem_util']:.1f}% (参考)")

        # ===== 分析模型 =====
        compute_type = "GPU 密集型 (含 GPU 能耗)" if is_gpu else "CPU 密集型"
        total_device_energy = stats['est_energy_mj']
        compute_energy = stats.get('compute_energy_mj', 0.0)
        memory_energy = stats.get('memory_energy_mj', 0.0)
        idle_energy = stats.get('idle_energy_mj', 0.0)

        # 判断是否使用了 Zeus
        is_zeus = "zeus" in est_method.lower()
        model_title = "Zeus 硬件计数器 + Analytical Model" if is_zeus else "Refined Analytical Model"

        lines.extend([
            "",
            f"  [设备能耗 — {model_title}]",
            f"  🔧 Compute type:        {compute_type}",
            f"  ⚡ Device TDP:          {stats['device_tdp_mw']:.0f} mW",
            f"  📐 Est. method:         {est_method}",
        ])

        if "gpu_name" in stats:
            lines.append(
                f"  🎮 GPU:                 {stats['gpu_name']}")
        if "gpu_energy_specs" in stats:
            lines.append(
                f"  📋 Energy specs:        {stats['gpu_energy_specs']}")
        if "idle_power_mw" in stats:
            lines.append(
                f"  💤 Idle power baseline: {stats['idle_power_mw']:.0f} mW")

        # 分析模型详情
        if "model_params_B" in stats:
            lines.extend([
                "",
                f"  [分析模型 — 每 token 操作量]",
                f"  🧠 Model:               {stats['model_params_B']:.2f}B params, "
                f"{stats['model_bytes_per_param']:.0f} bytes/param",
                f"  📐 FLOPs/token:         {stats['flops_per_token']:.2e} "
                f"(2 × P)",
            ])

            # 精细化 memory breakdown (方案 A)
            if "weight_bytes_per_tok" in stats:
                w_b = stats["weight_bytes_per_tok"]
                kv_r = stats["kv_read_bytes_per_tok"]
                kv_w = stats["kv_write_bytes_per_tok"]
                act_b = stats["activation_bytes_per_tok"]
                tot_b = stats["total_bytes_per_tok"]
                avg_sl = stats.get("avg_seq_len", 0)

                lines.extend([
                    f"  📐 Bytes/token (HBM):   {tot_b:.2e} (avg_seq_len={avg_sl})",
                    f"      ① 权重搬运:         {w_b:.2e} "
                    f"({w_b/1e6:.1f} MB, P×b)",
                    f"      ② KV cache 读:      {kv_r:.2e} "
                    f"({kv_r/1e6:.1f} MB, 2·L·d_kv·seq·b)",
                    f"      ③ KV cache 写:      {kv_w:.2e} "
                    f"({kv_w/1e3:.1f} KB, 2·L·d_kv·1·b)",
                    f"      ④ 激活值搬运:       {act_b:.2e} "
                    f"({act_b/1e3:.1f} KB, 2L(d+d_ff)b)",
                ])
            else:
                # 粗糙模式 (无 model.config)
                tot_b = stats.get("total_bytes_per_tok",
                                  stats.get("total_memory_bytes", 0) /
                                  max(tokens_generated, 1))
                lines.append(
                    f"  📐 Bytes/token (HBM):   {tot_b:.2e} (仅权重, 无 config)")

            if "theory_compute_mj" in stats:
                lines.extend([
                    f"  📐 Theory compute:      {stats['theory_compute_mj']:.3f} mJ "
                    f"(FLOPs × {stats['pj_per_flop']} pJ/FLOP)",
                    f"  📐 Theory memory:       {stats['theory_memory_mj']:.3f} mJ "
                    f"(Bytes × {stats['pj_per_byte']} pJ/Byte)",
                    f"  📐 Theory ratio:        "
                    f"compute {stats['compute_ratio']:.1%} : "
                    f"memory {stats['memory_ratio']:.1%}",
                ])

        # ===== 能耗拆分 =====
        lines.extend([
            "",
            f"  [能耗拆分 — 总能耗 = compute + memory + idle]",
            f"  ⚙️  Compute energy (ALU): {compute_energy:.0f} mJ "
            f"({compute_energy/1000:.3f} J)",
            f"  💿 Memory energy (HBM):  {memory_energy:.0f} mJ "
            f"({memory_energy/1000:.3f} J)",
        ])
        if idle_energy > 0:
            lines.append(
                f"  💤 Idle energy (static): {idle_energy:.0f} mJ "
                f"({idle_energy/1000:.3f} J)")
        lines.extend([
            f"  ─────────────────────────────────────",
            f"  🔋 Device total energy:  {total_device_energy:.0f} mJ "
            f"({total_device_energy/1000:.2f} J)",
            f"  🔋 Device avg power:     {stats['est_avg_power_mw']:.0f} mW",
        ])

        # 占比
        if total_device_energy > 0 and (compute_energy > 0 or memory_energy > 0):
            comp_pct = compute_energy / total_device_energy * 100
            mem_pct = memory_energy / total_device_energy * 100
            parts_str = f"compute {comp_pct:.1f}% | memory {mem_pct:.1f}%"
            if idle_energy > 0:
                idle_pct = idle_energy / total_device_energy * 100
                parts_str += f" | idle {idle_pct:.1f}%"
            lines.append(f"  📊 Breakdown:            {parts_str}")

        # ===== 网络能耗 =====
        net_energy = 0.0
        if net_stats and net_stats.get("net_total_bytes", 0) > 0:
            net_energy = net_stats["net_total_energy_mj"]
            tx_kb = net_stats["net_tx_bytes"] / 1024
            rx_kb = net_stats["net_rx_bytes"] / 1024
            lines.extend([
                "",
                f"  [网络传输能耗 — {net_stats['net_type'].upper()}]",
                f"  📤 TX (UAV→BS):        {tx_kb:.1f} KB "
                f"→ {net_stats['net_tx_energy_mj']:.1f} mJ",
                f"  📥 RX (BS→UAV):        {rx_kb:.1f} KB "
                f"→ {net_stats['net_rx_energy_mj']:.1f} mJ",
                f"  🔋 Network energy:      {net_energy:.1f} mJ",
            ])

        # ===== 汇总 =====
        total_energy = total_device_energy + net_energy
        lines.extend([
            "",
            f"  [汇总]",
            f"  🔋 Total energy:        {total_energy:.0f} mJ ({total_energy/1000:.2f} J)",
        ])
        if net_energy > 0 or (compute_energy > 0 and memory_energy > 0):
            parts = []
            if compute_energy > 0:
                parts.append(f"compute {compute_energy:.0f}")
            if memory_energy > 0:
                parts.append(f"memory {memory_energy:.0f}")
            if idle_energy > 0:
                parts.append(f"idle {idle_energy:.0f}")
            if net_energy > 0:
                parts.append(f"network {net_energy:.1f}")
            lines.append(f"     ({' + '.join(parts)} mJ)")

        if tokens_generated > 0:
            e_per_tok = total_energy / tokens_generated
            lines.append(f"  🔋 Energy/token:        {e_per_tok:.1f} mJ/token")
            if compute_energy > 0 and memory_energy > 0:
                lines.append(
                    f"     (compute {compute_energy/tokens_generated:.1f} + "
                    f"memory {memory_energy/tokens_generated:.1f}"
                    + (f" + idle {idle_energy/tokens_generated:.1f}"
                       if idle_energy > 0 else "")
                    + f" mJ/token)")

        # ===== 硬件级参考 =====
        hw = stats.get("sys_hw_backend", "none")
        if hw != "none" and "sys_avg_power_mw" in stats:
            lines.extend([
                "",
                f"  [硬件级参考 — 全系统, 可能含其他应用 ({hw})]",
                f"  ⚡ Sys avg power:       {stats['sys_avg_power_mw']:.0f} mW",
                f"  ⚡ Sys peak power:      {stats['sys_peak_power_mw']:.0f} mW",
                f"  🔋 Sys total energy:    {stats['sys_energy_mj']:.0f} mJ "
                f"({stats['sys_energy_mj']/1000:.2f} J)",
            ])

        if tokens_generated > 0:
            tps = tokens_generated / stats["wall_time"] if stats["wall_time"] > 0 else 0
            lines.append(f"\n  🚀 Throughput:          {tps:.2f} tokens/s")

        return "\n".join(lines)


# ============ TokenEnergyTracker ============

class TokenEnergyTracker:
    """
    逐 token 能耗记录器 —— 基于 Zeus 硬件能耗计数器。

    在每个 token 生成的前后各调用 begin_token() / end_token(),
    利用 Zeus 的 begin_window / end_window 测量该 token 的 GPU 能耗 (mJ)。

    多次推理后, 按 token 位置取平均, 得到 "每个位置平均能耗" 曲线。

    用法:
        tracker = TokenEnergyTracker(gpu_indices=[0])
        for sample in samples:
            tracker.new_sequence()
            for pos in range(max_tokens):
                tracker.begin_token(pos)
                # ... 生成一个 token ...
                tracker.end_token(pos)
        summary = tracker.summarize()
        # summary = {"per_position_mean_mj": [...], "per_position_std_mj": [...], ...}
    """

    def __init__(self, gpu_indices: Optional[List[int]] = None):
        """
        Args:
            gpu_indices: GPU 索引列表, e.g. [0]. 如果不传, 默认 [0]。
        """
        if gpu_indices is None:
            gpu_indices = [0]
        self._gpu_indices = gpu_indices

        # 初始化 Zeus (启用 approx_instant_energy 以支持短窗口测量)
        self._zeus_monitor = None
        self._active_window: Optional[str] = None
        try:
            from zeus.monitor.energy import ZeusMonitor
            from zeus.device.gpu import get_gpus

            # approx_instant_energy=True: 当硬件能耗计数器更新周期 > 测量窗口时,
            # 用 瞬时功率 × 窗口时长 近似, 避免返回 0
            zeus_mon = ZeusMonitor(
                gpu_indices=gpu_indices,
                approx_instant_energy=True,
            )
            gpus_mgr = get_gpus()
            gpu_obj = gpus_mgr._gpus[gpu_indices[0]]
            if gpu_obj.supportsGetTotalEnergyConsumption():
                self._zeus_monitor = zeus_mon
                print(f"[TokenEnergyTracker] Zeus 已就绪 (approx_instant_energy=True), "
                      f"监控 GPU: {gpu_indices}")
            else:
                print("[TokenEnergyTracker] Zeus: GPU 不支持硬件能耗计数器, 退回 time-based")
        except ImportError:
            pass
        except Exception as e:
            print(f"[TokenEnergyTracker] Zeus 初始化失败: {e}")

        if self._zeus_monitor is None:
            # 退回方案: 用 pynvml 瞬时功率 × Δt 估算 (所有 GPU)
            self._nvml_handles = []
            try:
                import pynvml
                pynvml.nvmlInit()
                for gi in gpu_indices:
                    self._nvml_handles.append(
                        pynvml.nvmlDeviceGetHandleByIndex(gi))
            except Exception:
                pass
        else:
            self._nvml_handles = []

        # 存储: list of list — 外层是 sequence, 内层是 (position, energy_mj)
        self._all_sequences: List[List[tuple]] = []
        self._current_seq: List[tuple] = []
        self._token_start_time: float = 0.0

    def new_sequence(self):
        """开始新的一条 sequence (新的 prompt)。"""
        if self._current_seq:
            self._all_sequences.append(self._current_seq)
        self._current_seq = []

    def begin_token(self, position: int):
        """
        在生成第 position 个 token 之前调用。

        Args:
            position: token 位置 (0-based)
        """
        self._token_start_time = time.time()
        if self._zeus_monitor is not None:
            window_name = f"tok_{id(self)}_{position}_{time.monotonic_ns()}"
            try:
                self._zeus_monitor.begin_window(window_name)
                self._active_window = window_name
            except Exception:
                self._active_window = None
        else:
            self._active_window = None

    def end_token(self, position: int):
        """
        在生成第 position 个 token 之后调用。
        记录该 token 的能耗 (mJ)。

        Args:
            position: token 位置 (0-based), 必须与 begin_token 匹配
        """
        dt = time.time() - self._token_start_time
        energy_mj = 0.0

        if self._zeus_monitor is not None and self._active_window is not None:
            try:
                result = self._zeus_monitor.end_window(self._active_window)
                # Zeus 返回 Joules → 转 mJ
                total_j = sum(result.gpu_energy.values()) if hasattr(
                    result, 'gpu_energy') else 0.0
                energy_mj = total_j * 1000.0
            except Exception:
                # Zeus 失败, 退回 time-based
                energy_mj = self._estimate_energy_time_based(dt)
        else:
            energy_mj = self._estimate_energy_time_based(dt)

        self._active_window = None
        self._current_seq.append((position, energy_mj))

    def _estimate_energy_time_based(self, dt: float) -> float:
        """
        退回方案: 用 pynvml 瞬时功率 × Δt 估算 (所有 GPU 求和)。

        Returns:
            能耗 (mJ)
        """
        if self._nvml_handles:
            try:
                import pynvml
                total_mj = 0.0
                for handle in self._nvml_handles:
                    # nvmlDeviceGetPowerUsage 返回 milliwatts
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    total_mj += power_mw * dt  # mW × s = mJ
                return total_mj
            except Exception:
                pass
        return 0.0

    def finish(self):
        """结束所有记录, 将最后一条 sequence 存入。"""
        if self._current_seq:
            self._all_sequences.append(self._current_seq)
            self._current_seq = []

    @property
    def num_sequences(self) -> int:
        """已记录的 sequence 数量。"""
        n = len(self._all_sequences)
        if self._current_seq:
            n += 1
        return n

    def summarize(self) -> dict:
        """
        按 token 位置取平均, 返回汇总字典。

        Returns:
            {
                "num_sequences": int,
                "max_position": int,
                "per_position_mean_mj": list[float],   # 每个位置的平均能耗 (mJ)
                "per_position_std_mj": list[float],     # 每个位置的标准差 (mJ)
                "per_position_count": list[int],         # 每个位置的样本数
                "per_position_min_mj": list[float],
                "per_position_max_mj": list[float],
                "total_mean_mj_per_token": float,       # 全部 token 的平均能耗
                "all_energies": list[list[tuple]],       # 原始数据 (可选保存)
            }
        """
        self.finish()

        if not self._all_sequences:
            return {
                "num_sequences": 0,
                "max_position": 0,
                "per_position_mean_mj": [],
                "per_position_std_mj": [],
                "per_position_count": [],
                "per_position_min_mj": [],
                "per_position_max_mj": [],
                "total_mean_mj_per_token": 0.0,
                "all_energies": [],
            }

        # 按 position 聚合
        from collections import defaultdict
        pos_energies = defaultdict(list)
        for seq in self._all_sequences:
            for pos, energy in seq:
                pos_energies[pos].append(energy)

        max_pos = max(pos_energies.keys()) if pos_energies else 0

        import math
        mean_list = []
        std_list = []
        count_list = []
        min_list = []
        max_list = []
        all_vals = []

        for p in range(max_pos + 1):
            vals = pos_energies.get(p, [])
            count_list.append(len(vals))
            if vals:
                m = sum(vals) / len(vals)
                mean_list.append(m)
                min_list.append(min(vals))
                max_list.append(max(vals))
                all_vals.extend(vals)
                if len(vals) > 1:
                    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
                    std_list.append(math.sqrt(var))
                else:
                    std_list.append(0.0)
            else:
                mean_list.append(0.0)
                std_list.append(0.0)
                min_list.append(0.0)
                max_list.append(0.0)

        total_mean = sum(all_vals) / len(all_vals) if all_vals else 0.0

        return {
            "num_sequences": len(self._all_sequences),
            "max_position": max_pos,
            "per_position_mean_mj": mean_list,
            "per_position_std_mj": std_list,
            "per_position_count": count_list,
            "per_position_min_mj": min_list,
            "per_position_max_mj": max_list,
            "total_mean_mj_per_token": total_mean,
            "all_energies": self._all_sequences,
        }
