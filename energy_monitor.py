"""
energy_monitor.py - UAV 端能耗监控模块

包含:
  - EnergyMonitor: 进程级能耗监控器（计算 + 网络）
  - 设备 TDP 常量表
  - 网络功耗 / 吞吐量常量表
"""

import time
import resource
import platform
import threading
import subprocess
from typing import Dict, Optional

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
                 framework: str = "pytorch", interval: float = 0.5):
        self.device = device
        self.framework = framework
        self.interval = interval

        # 判断是否为 GPU 密集型任务
        self._is_gpu_intensive = (
            framework == "mlx"
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

        # CUDA 相关
        self._nvml_handle = None

        # 设备 TDP
        self._device_tdp_mw = self._get_device_tdp()

        # 硬件级功率后端 (可选参考)
        self._hw_backend = self._detect_hw_backend()

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
        import pynvml
        power_mw = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
        return 0, power_mw

    def _sample_powermetrics(self):
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
            return cpu_mw, gpu_mw + ane_mw
        except Exception:
            return 0, 0

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
                cpu_mw, gpu_mw = sample_fn()
                self._power_samples.append((time.time(), cpu_mw, gpu_mw))
            except Exception:
                pass
            time.sleep(self.interval)

    # ---------- 公共 API ----------

    def start(self):
        """开始能耗监控"""
        self._power_samples = []
        self._start_time = time.time()
        self._rusage_start = resource.getrusage(resource.RUSAGE_SELF)

        if self._hw_backend != "none":
            self._running = True
            self._thread = threading.Thread(target=self._sampling_loop, daemon=True)
            self._thread.start()

        if self.device is not None and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def stop(self) -> Dict[str, float]:
        """停止监控并返回能耗统计字典"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None

        self._end_time = time.time()
        self._rusage_end = resource.getrusage(resource.RUSAGE_SELF)

        wall_time = self._end_time - self._start_time

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

        # 进程级能耗估算
        if self._is_gpu_intensive:
            if platform.system() == "Darwin":
                gpu_load_factor = 0.70
            else:
                gpu_load_factor = 0.85
            est_energy_mj = wall_time * self._device_tdp_mw * gpu_load_factor
            est_method = "gpu (wall_time × TDP × load_factor)"
        else:
            est_energy_mj = cpu_total * self._device_tdp_mw
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
            "est_energy_mj": round(est_energy_mj, 1),
            "est_avg_power_mw": round(est_avg_power_mw, 1),
        }

        # CUDA 峰值显存
        if self.device is not None and self.device.type == "cuda":
            peak_gpu_mem = torch.cuda.max_memory_allocated(self.device) / (1024**2)
            stats["peak_gpu_memory_mb"] = round(peak_gpu_mem, 1)

        # 硬件级参考数据
        stats["sys_hw_backend"] = self._hw_backend

        if self._power_samples:
            cpu_vals = [s[1] for s in self._power_samples]
            gpu_vals = [s[2] for s in self._power_samples]
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
        """格式化输出能耗报告"""
        is_gpu = stats.get("gpu_intensive", False)
        est_method = stats.get("est_method", "unknown")

        lines = [
            "",
            "  [进程级指标 — 仅本进程，不受其他应用影响]",
            f"  ⏱  Wall time:          {stats['wall_time']:.2f}s",
            f"  🖥  CPU time (process): {stats['cpu_total_time']:.2f}s "
            f"(user {stats['cpu_user_time']:.2f}s + sys {stats['cpu_sys_time']:.2f}s)",
            f"  📈 CPU utilization:     {stats['cpu_utilization']:.1f}%",
            f"  💾 Peak memory (RSS):   {stats['peak_memory_mb']:.1f} MB",
        ]

        if "peak_gpu_memory_mb" in stats:
            lines.append(
                f"  🎮 Peak GPU memory:     {stats['peak_gpu_memory_mb']:.1f} MB")

        compute_type = "GPU 密集型 (含 GPU 能耗)" if is_gpu else "CPU 密集型"
        compute_energy = stats['est_energy_mj']

        lines.extend([
            f"  🔧 Compute type:        {compute_type}",
            f"  ⚡ Device TDP:          {stats['device_tdp_mw']:.0f} mW",
            f"  📐 Est. method:         {est_method}",
            f"  🔋 Compute energy:      {compute_energy:.0f} mJ "
            f"({compute_energy/1000:.2f} J)",
            f"  🔋 Compute avg power:   {stats['est_avg_power_mw']:.0f} mW",
        ])

        # 网络能耗
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

        # 汇总
        total_energy = compute_energy + net_energy
        lines.extend([
            "",
            f"  [汇总]",
            f"  🔋 Total energy:        {total_energy:.0f} mJ ({total_energy/1000:.2f} J)",
        ])
        if net_energy > 0:
            lines.append(
                f"     (compute {compute_energy:.0f} + network {net_energy:.1f})")

        if tokens_generated > 0:
            e_per_tok = total_energy / tokens_generated
            lines.append(f"  🔋 Energy/token:        {e_per_tok:.1f} mJ/token")

        # 硬件级参考
        hw = stats.get("sys_hw_backend", "none")
        if hw != "none" and "sys_avg_power_mw" in stats:
            lines.extend([
                "",
                f"  [硬件级参考 — 全系统，可能含其他应用 ({hw})]",
                f"  ⚡ Sys avg power:       {stats['sys_avg_power_mw']:.0f} mW",
                f"  ⚡ Sys peak power:      {stats['sys_peak_power_mw']:.0f} mW",
                f"  🔋 Sys total energy:    {stats['sys_energy_mj']:.0f} mJ "
                f"({stats['sys_energy_mj']/1000:.2f} J)",
            ])

        if tokens_generated > 0:
            tps = tokens_generated / stats["wall_time"] if stats["wall_time"] > 0 else 0
            lines.append(f"\n  🚀 Throughput:          {tps:.2f} tokens/s")

        return "\n".join(lines)
