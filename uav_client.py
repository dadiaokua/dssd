"""
uav_client.py - UAV 端 (设备/客户端) 入口

在边缘设备上运行，加载小模型做 draft，通过 TCP 调用 BS 端做 verify。
本文件只包含 CLI 参数解析和主流程编排，具体逻辑分布在:
  - draft_node.py:     DraftNode 类（PyTorch / MLX）
  - decoding.py:       解码主循环（DSD / DSSD / Baseline）
  - energy_monitor.py: 能耗监控
  - dssd_net.py:       TCP 通信层
  - dssd_utils.py:     设备检测 / 采样工具

用法:

  Linux + CUDA:
    python uav_client.py \\
        --draft_model_name ./LLM/opt-125m \\
        --device cuda:0 \\
        --bs_addr 192.168.1.100

  Mac (Apple Silicon, 自动使用 MLX 加速):
    python uav_client.py \\
        --draft_model_name /path/to/Qwen3-0.6B \\
        --device auto \\
        --bs_addr 192.168.1.100

  仅本地 baseline (不需要 BS 服务器):
    python uav_client.py --mode local_baseline --device auto

依赖:
  - 通用: torch, transformers, tqdm
  - Mac 加速 (可选): mlx, mlx-lm  (pip install mlx-lm)
"""

import argparse

from dssd_net import UAVClient
from draft_node import create_draft_node
from decoding import (
    generate_DSD,
    generate_DSSD,
    baseline_autoregressive,
    baseline_local_autoregressive,
    run_benchmark,
    save_results,
    BENCHMARK_PROMPTS,
)
from network_shaper import NetworkShaper, NETWORK_PROFILES


def main():
    parser = argparse.ArgumentParser(description="UAV (Device) Client")
    parser.add_argument('--input', type=str,
                        default="Alan Turing theorized that computers would one day become ")
    parser.add_argument('--draft_model_name', type=str,
                        default="/Users/myrick/modelHub/Qwen3-1.7B")
    parser.add_argument('--device', type=str, default="auto",
                        help="Device: 'auto' (cuda>mps>cpu), 'cuda:0', 'mps', 'cpu'")
    parser.add_argument('--framework', type=str, default="auto",
                        choices=["auto", "mlx", "pytorch"],
                        help="Inference framework: 'auto' (Apple Silicon→MLX, CUDA→PyTorch), "
                             "'mlx' (force MLX), 'pytorch' (force PyTorch)")
    parser.add_argument('--bs_addr', type=str, default="127.0.0.1",
                        help="BS server IP address")
    parser.add_argument('--bs_port', type=int, default=50051,
                        help="BS server port")
    parser.add_argument('--max_len', type=int, default=256,
                        help="Maximum number of tokens to generate per request (default: 256)")
    parser.add_argument('--gamma', type=int, default=4)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--top_p', type=float, default=0)
    parser.add_argument('--mode', type=str, default="all",
                        choices=["dsd", "dssd", "baseline", "local_baseline",
                                 "all", "benchmark"],
                        help="Which mode to run: dsd, dssd, baseline (remote LLM), "
                             "local_baseline (local SLM), all (single run), "
                             "or benchmark (multi-prompt multi-trial)")
    parser.add_argument('--csv_path', type=str, default="results_real_network.csv",
                        help="Path to save results CSV")
    parser.add_argument('--net_type', type=str, default="wifi",
                        choices=["wifi", "lte", "eth"],
                        help="Network type for energy estimation: wifi, lte, eth")
    # ---------- Benchmark 参数 ----------
    parser.add_argument('--num_trials', type=int, default=3,
                        help="Number of repeated trials per prompt in benchmark mode (default: 3)")
    parser.add_argument('--num_prompts', type=int, default=0,
                        help="Number of prompts from built-in set (0 = use all, default: 0)")
    parser.add_argument('--bench_modes', type=str, default=None,
                        help="Comma-separated methods for benchmark, e.g. 'dssd,dsd,local_baseline'. "
                             "Default: auto (all if BS connected, else local_baseline only)")
    # ---------- 网络限速 (Traffic Shaping) ----------
    parser.add_argument('--tc_enable', action='store_true',
                        help="Enable OS-level traffic shaping (requires sudo)")
    parser.add_argument('--tc_profile', type=str, default=None,
                        choices=list(NETWORK_PROFILES.keys()),
                        help="Use a preset network profile (overrides tc_bw/tc_delay/tc_jitter/tc_loss). "
                             "Run 'python network_shaper.py list' to see all profiles.")
    parser.add_argument('--tc_bw', type=str, default="10mbit",
                        help="Bandwidth limit (e.g. '1mbit', '500kbit', '50mbit'). Default: 10mbit")
    parser.add_argument('--tc_delay', type=str, default="50ms",
                        help="One-way delay (RTT ≈ 2x). e.g. '50ms', '100ms'. Default: 50ms")
    parser.add_argument('--tc_jitter', type=str, default="10ms",
                        help="Delay jitter range (e.g. '20ms' = delay ± 20ms). Default: 10ms")
    parser.add_argument('--tc_loss', type=str, default="0%%",
                        help="Packet loss rate (e.g. '1%%', '5%%'). Default: 0%%")
    parser.add_argument('--tc_list_profiles', action='store_true',
                        help="List available network profiles and exit")
    args = parser.parse_args()

    # 如果只是列出 profiles 则打印后退出
    if args.tc_list_profiles:
        NetworkShaper.list_profiles()
        return

    # 根据设备自动选择框架，加载小模型
    uav_node, tokenizer = create_draft_node(
        model_name=args.draft_model_name,
        device_str=args.device,
        framework=args.framework,
        args=args,
    )

    results = []

    # ---------- 网络限速 ----------
    shaper = None
    if args.tc_enable:
        shaper = NetworkShaper(
            bandwidth=args.tc_bw,
            delay=args.tc_delay,
            jitter=args.tc_jitter,
            loss=args.tc_loss,
            target_port=args.bs_port,
            profile=args.tc_profile,
        )
        shaper.apply()

    # ---------- 需要 BS 连接的模式 ----------
    need_bs = args.mode in ("dssd", "dsd", "baseline", "all", "benchmark")

    if need_bs:
        # benchmark 模式下如果只跑 local_baseline 则不需要连接
        if args.mode == "benchmark" and args.bench_modes:
            bench_mode_list = [m.strip() for m in args.bench_modes.split(",")]
            need_bs = any(m in ("dssd", "dsd", "baseline") for m in bench_mode_list)

    if need_bs:
        client = UAVClient(bs_host=args.bs_addr, bs_port=args.bs_port)
        client.connect()
    else:
        client = None

    try:
        # ==================== Benchmark 模式 ====================
        if args.mode == "benchmark":
            # 选择 prompts
            prompts = BENCHMARK_PROMPTS
            if args.num_prompts > 0:
                prompts = prompts[:args.num_prompts]

            # 选择 modes
            bench_modes = None
            if args.bench_modes:
                bench_modes = [m.strip() for m in args.bench_modes.split(",")]

            all_raw, summaries = run_benchmark(
                uav_node=uav_node,
                client=client,
                tokenizer=tokenizer,
                args=args,
                prompts=prompts,
                num_trials=args.num_trials,
                modes=bench_modes,
                tc_config=shaper.get_config() if shaper else None,
            )
            # 原始结果已在 run_benchmark 中逐条实时写入 *_raw.csv

            # 保存汇总统计
            if summaries:
                summary_csv = args.csv_path.replace(".csv", "_summary.csv")
                save_results(summaries, summary_csv)

        # ==================== 单次模式 ====================
        else:
            input_ids = tokenizer.encode(args.input, return_tensors='pt')

            if args.mode in ("dssd", "all"):
                print("\n" + "=" * 60)
                print(" Running DSSD (Distributed Split Speculative Decoding)")
                print("=" * 60)
                r = generate_DSSD(uav_node, client, input_ids, tokenizer, args)
                results.append(r)

            if args.mode in ("dsd", "all"):
                print("\n" + "=" * 60)
                print(" Running DSD (Distributed Speculative Decoding)")
                print("=" * 60)
                r = generate_DSD(uav_node, client, input_ids, tokenizer, args)
                results.append(r)

            if args.mode in ("baseline", "all"):
                print("\n" + "=" * 60)
                print(" Running Baseline (Remote LLM Autoregressive)")
                print("=" * 60)
                r = baseline_autoregressive(client, input_ids, tokenizer, args)
                results.append(r)

            if args.mode in ("local_baseline", "all"):
                print("\n" + "=" * 60)
                print(" Running Baseline (Local SLM Autoregressive)")
                print("=" * 60)
                r = baseline_local_autoregressive(uav_node, input_ids, tokenizer, args)
                results.append(r)

            # 将网络限速配置写入每条结果
            if shaper is not None:
                tc_cfg = shaper.get_config()
                for r in results:
                    r.update(tc_cfg)

            if results:
                save_results(results, args.csv_path)

    finally:
        if client is not None:
            client.close()
        if shaper is not None:
            shaper.remove()
        print("\n[UAV Client] Done.")


if __name__ == "__main__":
    main()
