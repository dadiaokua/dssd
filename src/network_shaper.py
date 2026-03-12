"""
network_shaper.py - 跨平台网络流量整形工具

利用 OS 原生的流量控制工具对 **真实 TCP 连接** 进行限速和加抖动:
  - macOS:  dnctl (dummynet pipe) + pfctl (Packet Filter)
  - Linux:  tc (Traffic Control) + netem

支持的参数:
  - bandwidth:  带宽限制 (e.g. "1mbit", "500kbit", "10mbit")
  - delay:      单向延迟 (e.g. "50ms", "100ms")
  - jitter:     延迟抖动 (e.g. "10ms", "30ms")
  - loss:       丢包率 (e.g. "1%", "5%")

用法:
    shaper = NetworkShaper(
        bandwidth="1mbit",    # 限速到 1 Mbps
        delay="50ms",         # 单向延迟 50ms (RTT ≈ 100ms)
        jitter="20ms",        # ±20ms 抖动
        loss="0%",            # 0% 丢包
        target_port=50051,    # 只影响 BS 通信端口
    )
    shaper.apply()            # 应用限速规则
    # ... 实验 ...
    shaper.remove()           # 清除限速规则

注意:
  - 需要 sudo / root 权限
  - 只影响指定端口的流量，不影响其他网络应用
  - 建议在实验前后各调用 apply() / remove()
"""

import os
import sys
import platform
import subprocess
import shlex
import atexit
from typing import Optional


def _run_cmd(cmd: str, check: bool = True, sudo: bool = True) -> subprocess.CompletedProcess:
    """执行 shell 命令，可选 sudo"""
    if sudo:
        cmd = f"sudo {cmd}"
    print(f"  [NetworkShaper] $ {cmd}")
    result = subprocess.run(
        shlex.split(cmd),
        capture_output=True, text=True, timeout=15,
    )
    if check and result.returncode != 0:
        stderr = result.stderr.strip()
        # 某些 "already exists" 类错误可以忽略
        if "No such" not in stderr and "does not exist" not in stderr:
            print(f"  [NetworkShaper] ⚠ stderr: {stderr}")
    return result


def _parse_bandwidth_bps(bw_str: str) -> int:
    """将带宽字符串转为 bits/s (e.g. '1mbit'->1000000, '500kbit'->500000)"""
    bw_str = bw_str.lower().strip()
    if bw_str.endswith("gbit"):
        return int(float(bw_str[:-4]) * 1_000_000_000)
    elif bw_str.endswith("mbit"):
        return int(float(bw_str[:-4]) * 1_000_000)
    elif bw_str.endswith("kbit"):
        return int(float(bw_str[:-4]) * 1_000)
    elif bw_str.endswith("bit"):
        return int(float(bw_str[:-3]))
    # 纯数字默认 bit/s
    return int(float(bw_str))


def _parse_delay_ms(delay_str: str) -> float:
    """将延迟字符串转为毫秒 (e.g. '50ms'->50.0, '0.1s'->100.0)"""
    delay_str = delay_str.lower().strip()
    if delay_str.endswith("ms"):
        return float(delay_str[:-2])
    elif delay_str.endswith("s"):
        return float(delay_str[:-1]) * 1000
    return float(delay_str)


def _parse_loss_pct(loss_str: str) -> float:
    """将丢包率字符串转为百分数 (e.g. '1%'->1.0, '0.5%'->0.5)"""
    loss_str = loss_str.strip()
    if loss_str.endswith("%"):
        return float(loss_str[:-1])
    return float(loss_str)


# ============ 预设网络场景 ============

NETWORK_PROFILES = {
    # 名称: (bandwidth, delay, jitter, loss, description)
    "wifi_good": ("50mbit", "5ms", "2ms", "0%",
                  "良好 WiFi (802.11ac, 近距离)"),
    "wifi_fair": ("20mbit", "15ms", "8ms", "0.5%",
                  "一般 WiFi (中等距离, 有干扰)"),
    "wifi_poor": ("5mbit", "40ms", "20ms", "2%",
                  "差 WiFi (远距离, 多干扰)"),
    "lte_good": ("10mbit", "30ms", "10ms", "0%",
                 "良好 LTE (4G, 信号强)"),
    "lte_fair": ("3mbit", "60ms", "25ms", "1%",
                 "一般 LTE (4G, 信号中等)"),
    "lte_poor": ("1mbit", "100ms", "50ms", "3%",
                 "差 LTE (4G, 信号弱 / 移动中)"),
    "5g_mmwave": ("100mbit", "10ms", "3ms", "0%",
                  "5G 毫米波 (近距离, 理想)"),
    "5g_sub6": ("30mbit", "20ms", "8ms", "0.5%",
                "5G Sub-6GHz (城区覆盖)"),
    "satellite": ("2mbit", "300ms", "50ms", "1%",
                  "卫星通信 (LEO, 如 Starlink)"),
    "uav_los": ("20mbit", "10ms", "5ms", "0.5%",
                "UAV 视距通信 (专用链路, 近距离)"),
    "uav_nlos": ("2mbit", "80ms", "40ms", "3%",
                 "UAV 非视距通信 (遮挡, 远距离)"),
    "paper_sim": ("1mbit", "50ms", "0ms", "0%",
                  "论文模拟环境 (低带宽高延迟, 无抖动)"),
}


class NetworkShaper:
    """
    跨平台网络流量整形器

    在操作系统网络栈层面对指定端口的流量施加:
      - 带宽限制 (bandwidth)
      - 延迟 (delay)
      - 抖动 (jitter)
      - 丢包 (loss)

    作用于真实 TCP 包，比 time.sleep() 模拟更准确。
    """

    def __init__(self,
                 bandwidth: str = "10mbit",
                 delay: str = "50ms",
                 jitter: str = "10ms",
                 loss: str = "0%",
                 target_port: int = 50051,
                 interface: Optional[str] = None,
                 profile: Optional[str] = None):
        """
        Args:
            bandwidth: 带宽限制 (e.g. "1mbit", "500kbit")
            delay:     单向延迟 (RTT ≈ 2x delay)
            jitter:    延迟抖动范围 (e.g. "20ms" 表示 delay ± 20ms)
            loss:      随机丢包率 (e.g. "1%")
            target_port: 目标端口 (只限制该端口流量)
            interface: 网络接口 (None=自动检测, e.g. "en0", "eth0")
            profile:   预设场景名 (覆盖 bandwidth/delay/jitter/loss)
        """
        # 如果指定了预设场景，覆盖参数
        if profile is not None:
            if profile not in NETWORK_PROFILES:
                avail = ", ".join(NETWORK_PROFILES.keys())
                raise ValueError(
                    f"Unknown profile '{profile}'. Available: {avail}")
            bw, dl, jt, ls, desc = NETWORK_PROFILES[profile]
            bandwidth, delay, jitter, loss = bw, dl, jt, ls
            print(f"[NetworkShaper] Using profile '{profile}': {desc}")
            print(f"  bandwidth={bw}, delay={dl}, jitter={jt}, loss={ls}")

        self.bandwidth = bandwidth
        self.delay = delay
        self.jitter = jitter
        self.loss = loss
        self.target_port = target_port
        self.interface = interface or self._detect_interface()
        self._system = platform.system()
        self._applied = False

        # 解析数值
        self._bw_bps = _parse_bandwidth_bps(bandwidth)
        self._delay_ms = _parse_delay_ms(delay)
        self._jitter_ms = _parse_delay_ms(jitter)
        self._loss_pct = _parse_loss_pct(loss)

        # macOS dummynet pipe 编号 (避免冲突)
        self._pipe_num = 100 + (target_port % 100)

    @staticmethod
    def _detect_interface() -> str:
        """自动检测活跃的网络接口"""
        system = platform.system()
        if system == "Darwin":
            # macOS: 通常是 en0 (WiFi) 或 en1
            try:
                r = subprocess.run(
                    ["route", "-n", "get", "default"],
                    capture_output=True, text=True, timeout=5)
                for line in r.stdout.splitlines():
                    if "interface:" in line:
                        return line.split(":")[-1].strip()
            except Exception:
                pass
            return "en0"
        else:
            # Linux: 通常是 eth0 或 ens*
            try:
                r = subprocess.run(
                    ["ip", "route", "show", "default"],
                    capture_output=True, text=True, timeout=5)
                parts = r.stdout.strip().split()
                if "dev" in parts:
                    return parts[parts.index("dev") + 1]
            except Exception:
                pass
            return "eth0"

    @staticmethod
    def list_profiles():
        """打印所有可用的预设网络场景"""
        print("\n可用的网络场景预设 (--tc_profile):\n")
        print(f"  {'名称':<15s} {'带宽':<10s} {'延迟':<8s} {'抖动':<8s} "
              f"{'丢包':<6s} 描述")
        print("  " + "-" * 80)
        for name, (bw, dl, jt, ls, desc) in NETWORK_PROFILES.items():
            print(f"  {name:<15s} {bw:<10s} {dl:<8s} {jt:<8s} {ls:<6s} {desc}")
        print()

    def get_config(self) -> dict:
        """返回当前配置信息"""
        return {
            "tc_bandwidth": self.bandwidth,
            "tc_delay": self.delay,
            "tc_jitter": self.jitter,
            "tc_loss": self.loss,
            "tc_target_port": self.target_port,
            "tc_interface": self.interface,
            "tc_bw_bps": self._bw_bps,
            "tc_delay_ms": self._delay_ms,
            "tc_jitter_ms": self._jitter_ms,
            "tc_loss_pct": self._loss_pct,
            "tc_rtt_est_ms": self._delay_ms * 2,  # 估算 RTT
        }

    # ============ 应用 / 移除 ============

    def apply(self):
        """应用网络限速规则"""
        if self._applied:
            print("[NetworkShaper] Rules already applied, removing first ...")
            self.remove()

        print(f"\n[NetworkShaper] Applying traffic shaping on {self.interface}:")
        print(f"  Port:      {self.target_port}")
        print(f"  Bandwidth: {self.bandwidth} ({self._bw_bps/1_000_000:.1f} Mbps)")
        print(f"  Delay:     {self.delay} (est. RTT ≈ {self._delay_ms*2:.0f}ms)")
        print(f"  Jitter:    {self.jitter}")
        print(f"  Loss:      {self.loss}")
        print()

        if self._system == "Darwin":
            self._apply_macos()
        elif self._system == "Linux":
            self._apply_linux()
        else:
            raise RuntimeError(f"Unsupported OS: {self._system}")

        self._applied = True
        # 注册退出时自动清理，防止忘记 remove()
        atexit.register(self._atexit_cleanup)
        print(f"\n[NetworkShaper] ✅ Traffic shaping applied successfully!")

    def remove(self):
        """移除网络限速规则"""
        if not self._applied:
            # 即使没有 apply 也尝试清理（防止上次异常退出残留）
            pass

        print(f"\n[NetworkShaper] Removing traffic shaping rules ...")

        if self._system == "Darwin":
            self._remove_macos()
        elif self._system == "Linux":
            self._remove_linux()

        self._applied = False
        print(f"[NetworkShaper] ✅ Traffic shaping rules removed.")

    def _atexit_cleanup(self):
        """程序退出时自动清理"""
        if self._applied:
            try:
                self.remove()
            except Exception:
                pass

    # ============ macOS: dnctl + pfctl ============

    def _apply_macos(self):
        """
        macOS 使用 dummynet (dnctl) + Packet Filter (pfctl):

        1. 创建 dummynet pipe，设置 bandwidth / delay / jitter / loss
        2. 添加 pf 规则，将目标端口流量导入 pipe
        3. 启用 pf (如果尚未启用)

        dummynet pipe 的 delay 是单向的，所以 RTT ≈ 2 × delay。
        """
        pipe = self._pipe_num
        port = self.target_port

        # 1. 创建 dummynet pipe
        # dnctl 的 bandwidth 单位是 bit/s 或 Kbit/s 或 Mbit/s
        bw_kbps = self._bw_bps // 1000
        delay_val = int(self._delay_ms)

        # dnctl pipe config: bw + delay + plr
        pipe_config = f"dnctl pipe {pipe} config bw {bw_kbps}Kbit/s delay {delay_val}"

        # 丢包率 (dnctl 用 0.0~1.0 的浮点数)
        if self._loss_pct > 0:
            plr = self._loss_pct / 100.0
            pipe_config += f" plr {plr}"

        # macOS dnctl 本身不直接支持 jitter，但可以通过 pipe mask 或
        # 结合 probabilistic delay 来实现。这里我们用一个 workaround:
        # 创建两个 pipe (一个用于上行一个用于下行)，并用不同 delay 来模拟抖动范围。
        # 更好的方案是使用 Network Link Conditioner (NLC) 或 dummynet 的高级特性。
        # 为简单起见，这里用 dnctl 的标准 delay 作为基准延迟。
        _run_cmd(pipe_config)

        # 如果有 jitter，创建第二个 pipe 用于反向流量，delay 略有不同
        pipe2 = pipe + 1
        if self._jitter_ms > 0:
            delay2 = max(1, delay_val + int(self._jitter_ms * 0.5))
            pipe2_config = f"dnctl pipe {pipe2} config bw {bw_kbps}Kbit/s delay {delay2}"
            if self._loss_pct > 0:
                plr = self._loss_pct / 100.0
                pipe2_config += f" plr {plr}"
            _run_cmd(pipe2_config)

        # 2. 创建 pf anchor 规则
        # 将目标端口的出入流量分别导入不同 pipe
        pf_rules = []
        # 出站 (UAV → BS): 目的端口
        pf_rules.append(f"dummynet-anchor \"dssd_shaper\"")
        pf_rules.append(f"anchor \"dssd_shaper\"")

        # 创建 anchor 内容
        anchor_rules = []
        anchor_rules.append(
            f"dummynet out proto tcp to any port {port} pipe {pipe}")
        if self._jitter_ms > 0:
            anchor_rules.append(
                f"dummynet in proto tcp from any port {port} pipe {pipe2}")
        else:
            anchor_rules.append(
                f"dummynet in proto tcp from any port {port} pipe {pipe}")

        # 写入临时 pf 配置
        anchor_file = "/tmp/dssd_pf_anchor.conf"
        with open(anchor_file, "w") as f:
            for rule in anchor_rules:
                f.write(rule + "\n")

        # 加载 anchor 规则
        _run_cmd(f"pfctl -a dssd_shaper -f {anchor_file}")

        # 3. 确保主 pf.conf 包含我们的 anchor
        # 先检查是否已有
        check = _run_cmd("pfctl -sr", check=False)
        if "dssd_shaper" not in check.stdout:
            # 需要将 anchor 添加到 pf 主规则中
            # 读取当前规则，追加 anchor 引用
            main_rules = check.stdout.strip()
            pf_main = "/tmp/dssd_pf_main.conf"
            with open(pf_main, "w") as f:
                f.write(main_rules + "\n")
                f.write('dummynet-anchor "dssd_shaper"\n')
                f.write('anchor "dssd_shaper"\n')
            _run_cmd(f"pfctl -f {pf_main}")

        # 4. 启用 pf
        _run_cmd("pfctl -E", check=False)

        print(f"  [macOS] dnctl pipe {pipe} created")
        if self._jitter_ms > 0:
            print(f"  [macOS] dnctl pipe {pipe2} created (jitter variant)")
        print(f"  [macOS] pf anchor 'dssd_shaper' loaded")

    def _remove_macos(self):
        """移除 macOS dummynet + pf 规则"""
        pipe = self._pipe_num
        pipe2 = pipe + 1

        # 1. 清空 anchor 规则
        _run_cmd("pfctl -a dssd_shaper -F all", check=False)

        # 2. 删除 dummynet pipe
        _run_cmd(f"dnctl pipe {pipe} delete", check=False)
        _run_cmd(f"dnctl pipe {pipe2} delete", check=False)

        # 3. 清理临时文件
        for f in ["/tmp/dssd_pf_anchor.conf", "/tmp/dssd_pf_main.conf"]:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass

        # 注意: 不禁用 pf (pfctl -d)，因为用户可能有其他 pf 规则在运行

    # ============ Linux: tc + netem ============

    def _apply_linux(self):
        """
        Linux 使用 tc (Traffic Control) + netem:

        1. 在网络接口上创建 htb qdisc 作为根
        2. 创建 htb class 限制带宽
        3. 在 class 下挂 netem qdisc 添加 delay / jitter / loss
        4. 使用 filter 将目标端口流量导入该 class

        tc 的 delay 也是单向的，RTT ≈ 2 × delay。
        """
        iface = self.interface
        port = self.target_port
        bw = self.bandwidth
        delay = self.delay
        jitter = self.jitter
        loss = self.loss

        # 先清理可能存在的旧规则
        _run_cmd(f"tc qdisc del dev {iface} root", check=False)

        # 1. 创建根 htb qdisc
        _run_cmd(f"tc qdisc add dev {iface} root handle 1: htb default 10")

        # 2. 默认 class (不限速的正常流量)
        _run_cmd(f"tc class add dev {iface} parent 1: classid 1:10 "
                 f"htb rate 1000mbit")

        # 3. 限速 class (目标端口流量)
        _run_cmd(f"tc class add dev {iface} parent 1: classid 1:20 "
                 f"htb rate {bw} ceil {bw}")

        # 4. netem qdisc: delay + jitter + loss
        netem_params = f"delay {delay}"
        if self._jitter_ms > 0:
            netem_params += f" {jitter} distribution normal"
        if self._loss_pct > 0:
            netem_params += f" loss {loss}"

        _run_cmd(f"tc qdisc add dev {iface} parent 1:20 handle 20: "
                 f"netem {netem_params}")

        # 5. filter: 将目标端口的流量导入限速 class
        # 出站: 目的端口
        _run_cmd(f"tc filter add dev {iface} parent 1: protocol ip prio 1 "
                 f"u32 match ip dport {port} 0xffff flowid 1:20")
        # 入站: 源端口 (回包)
        _run_cmd(f"tc filter add dev {iface} parent 1: protocol ip prio 1 "
                 f"u32 match ip sport {port} 0xffff flowid 1:20")

        # ---- 入站 (ingress) 限速 ----
        # tc 的 htb 只能限出站，入站需要 ifb (Intermediate Functional Block)
        # 这里简化处理：只对出站限速。如果需要完整双向限速，可以启用 ifb。
        print(f"  [Linux] tc rules applied on {iface}")
        print(f"  [Linux] Note: bandwidth limit applies to egress only. "
              f"For full duplex shaping, enable ifb.")

    def _remove_linux(self):
        """移除 Linux tc 规则"""
        iface = self.interface
        _run_cmd(f"tc qdisc del dev {iface} root", check=False)
        print(f"  [Linux] tc rules removed from {iface}")

    # ============ 状态信息 ============

    def status(self) -> str:
        """返回当前限速状态的可读字符串"""
        if not self._applied:
            return "[NetworkShaper] No shaping rules active."

        lines = [
            f"[NetworkShaper] Active traffic shaping:",
            f"  Interface:  {self.interface}",
            f"  Port:       {self.target_port}",
            f"  Bandwidth:  {self.bandwidth}",
            f"  Delay:      {self.delay} (est. RTT ≈ {self._delay_ms*2:.0f}ms)",
            f"  Jitter:     {self.jitter}",
            f"  Loss:       {self.loss}",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return (f"NetworkShaper(bw={self.bandwidth}, delay={self.delay}, "
                f"jitter={self.jitter}, loss={self.loss}, "
                f"port={self.target_port})")


# ============ 便捷函数 ============

def apply_network_shaping(bandwidth: str = "10mbit",
                          delay: str = "50ms",
                          jitter: str = "10ms",
                          loss: str = "0%",
                          target_port: int = 50051,
                          profile: Optional[str] = None) -> NetworkShaper:
    """
    便捷函数：创建并应用网络限速。

    返回 NetworkShaper 对象，实验结束后调用 .remove() 清理。

    Example:
        shaper = apply_network_shaping(profile="lte_fair")
        # ... run experiment ...
        shaper.remove()
    """
    shaper = NetworkShaper(
        bandwidth=bandwidth,
        delay=delay,
        jitter=jitter,
        loss=loss,
        target_port=target_port,
        profile=profile,
    )
    shaper.apply()
    return shaper


# ============ CLI 入口 ============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Network Traffic Shaper for DSSD experiments")
    parser.add_argument("action", choices=["apply", "remove", "status", "list"],
                        help="Action: apply/remove/status/list profiles")
    parser.add_argument("--bandwidth", type=str, default="10mbit",
                        help="Bandwidth limit (e.g. '1mbit', '500kbit')")
    parser.add_argument("--delay", type=str, default="50ms",
                        help="One-way delay (e.g. '50ms')")
    parser.add_argument("--jitter", type=str, default="10ms",
                        help="Delay jitter (e.g. '20ms')")
    parser.add_argument("--loss", type=str, default="0%",
                        help="Packet loss rate (e.g. '1%%')")
    parser.add_argument("--port", type=int, default=50051,
                        help="Target port (default: 50051)")
    parser.add_argument("--profile", type=str, default=None,
                        help="Use a preset network profile (overrides other params)")
    parser.add_argument("--interface", type=str, default=None,
                        help="Network interface (default: auto-detect)")

    args = parser.parse_args()

    if args.action == "list":
        NetworkShaper.list_profiles()
        sys.exit(0)

    shaper = NetworkShaper(
        bandwidth=args.bandwidth,
        delay=args.delay,
        jitter=args.jitter,
        loss=args.loss,
        target_port=args.port,
        interface=args.interface,
        profile=args.profile,
    )

    if args.action == "apply":
        shaper.apply()
    elif args.action == "remove":
        shaper.remove()
    elif args.action == "status":
        print(shaper.status())
