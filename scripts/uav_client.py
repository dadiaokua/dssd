"""
uav_client.py - UAV 端 (设备/客户端) 入口

在边缘设备上运行，加载小模型做 draft，通过 TCP 调用 BS 端做 verify。
本文件只包含 CLI 参数解析和主流程编排，具体逻辑分布在 src/ 目录:
  - src/draft_node.py:     DraftNode 类（PyTorch / vLLM / MLX）
  - src/decoding.py:       解码主循环（DSD / DSSD / Baseline）
  - src/energy_monitor.py: 能耗监控 (Zeus + 解析模型)
  - src/dssd_net.py:       TCP 通信层
  - src/dssd_utils.py:     设备检测 / 采样工具
  - src/dataset_loader.py: 多数据集加载器

用法 (从项目根目录运行):

  Linux + CUDA:
    python scripts/uav_client.py \\
        --draft_model_name ./LLM/opt-125m \\
        --device cuda:0 \\
        --bs_addr 192.168.1.100

  Mac (Apple Silicon, 自动使用 MLX 加速):
    python scripts/uav_client.py \\
        --draft_model_name /path/to/Qwen3-0.6B \\
        --device auto \\
        --bs_addr 192.168.1.100

  仅本地 baseline (不需要 BS 服务器):
    python scripts/uav_client.py --mode local_baseline --device auto

  逐 token 能耗测量:
    python scripts/uav_client.py --mode token_energy --device auto

依赖:
  - 通用: torch, transformers, tqdm
  - Mac 加速 (可选): mlx, mlx-lm  (pip install mlx-lm)
  - 能耗监控: zeus-ml, pynvml
"""

import argparse
import os
import sys

# 将 src/ 加入模块搜索路径
_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from dssd_net import UAVClient
from draft_node import create_draft_node
from decoding import (
    generate_DSD,
    generate_DSSD,
    baseline_autoregressive,
    baseline_local_autoregressive,
    run_benchmark,
    run_kv_cache_benchmark,
    run_token_energy_benchmark,
    run_token_energy_batch_benchmark,
    run_token_energy_stream_benchmark,
    save_results,
    BENCHMARK_PROMPTS,
)
from network_shaper import NetworkShaper, NETWORK_PROFILES
from visualize_token_energy import run_visualization


# ============ 可用的对比方法 ============

_ALL_METHODS = [
    ("dssd",           "DSSD (Distributed Split Speculative Decoding)"),
    ("dsd",            "DSD  (Distributed Speculative Decoding)"),
    ("baseline",       "Baseline — Remote LLM Autoregressive"),
    ("local_baseline", "Baseline — Local SLM Autoregressive"),
]


def _interactive_method_selection(args):
    """
    交互式方法选择菜单。
    用户输入数字 (逗号分隔) 选择要运行的方法, 例如 "1,4" 或 "1 4"。
    输入 0 或 'all' 选择全部。
    """
    # 如果命令行已经通过 --bench_modes 指定了, 跳过交互
    if args.bench_modes:
        return args

    print("\n" + "=" * 60)
    print("  选择要运行的对比方法 (Select methods to run)")
    print("=" * 60)
    print(f"  0. [全部] All methods")
    for i, (key, desc) in enumerate(_ALL_METHODS, start=1):
        print(f"  {i}. [{key}] {desc}")
    print("-" * 60)
    print("  输入方法编号, 用逗号或空格分隔 (e.g. '1,4' or '1 4')")
    print("  输入 0 选择全部, 直接回车默认全部")
    print("-" * 60)

    try:
        user_input = input("  >>> ").strip()
    except (EOFError, KeyboardInterrupt):
        user_input = ""

    if not user_input or user_input == "0" or user_input.lower() == "all":
        print("  ✅ 已选择: 全部方法")
        return args

    # 解析数字: 支持 "1,4" "1 4" "1, 4" 等格式
    import re
    nums = re.findall(r'\d+', user_input)
    selected = []
    for n_str in nums:
        n = int(n_str)
        if n == 0:
            # 选了 0 就是全部
            print("  ✅ 已选择: 全部方法")
            return args
        if 1 <= n <= len(_ALL_METHODS):
            method_key = _ALL_METHODS[n - 1][0]
            if method_key not in selected:
                selected.append(method_key)

    if not selected:
        print("  ⚠ 无有效选择, 使用全部方法")
        return args

    selected_desc = ", ".join(selected)
    print(f"  ✅ 已选择: {selected_desc}")

    args.bench_modes = ",".join(selected)

    # 如果用户选了单个方法且不是 benchmark/kv_benchmark 模式, 直接设为该模式
    if len(selected) == 1 and args.mode not in ("benchmark", "kv_benchmark"):
        args.mode = selected[0]

    return args


def _make_experiment_dir(args):
    """
    根据实验配置和当前时间, 在 output/ 下创建独立的实验目录。

    目录命名格式:
        <mode>_<model_short>_<engine>_<key_params>_<YYYYMMDD_HHmmss>

    例如:
        token_energy_batch_Qwen3-32B_vllm_n20_t15000_r5_20260311_143052
        token_energy_stream_Qwen3-32B_vllm_n50_t512_rate10_dur600_20260311_150000
        kv_benchmark_Qwen3-32B_vllm_kv32-1024_20260311_160000
        benchmark_opt-125m_pytorch_20260311_170000

    同时更新 args.csv_path 指向该目录。
    """
    from datetime import datetime

    # 模型短名 (取最后一段路径)
    model_short = os.path.basename(args.draft_model_name.rstrip("/"))

    # 引擎
    engine = getattr(args, 'engine', 'auto')

    # 时间戳 (24 小时制)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 根据 mode 构建参数标签
    mode = args.mode
    param_parts = []

    if mode in ("token_energy", "token_energy_batch", "token_energy_stream"):
        param_parts.append(f"n{args.token_samples}")
        param_parts.append(f"t{args.token_max_tokens}")
        if mode == "token_energy_batch" and args.batch_repeats > 1:
            param_parts.append(f"r{args.batch_repeats}")
        if mode == "token_energy_stream":
            param_parts.append(f"rate{args.req_rate:.0f}")
            param_parts.append(f"dur{args.duration}")
            if args.warmup > 0:
                param_parts.append(f"w{args.warmup}")
            if args.batch_repeats > 1:
                param_parts.append(f"r{args.batch_repeats}")
    elif mode == "kv_benchmark":
        kv_str = args.kv_lengths.replace(",", "-")
        param_parts.append(f"kv{kv_str}")
        param_parts.append(f"g{args.gen_tokens}")
    elif mode == "benchmark":
        param_parts.append(f"t{args.num_trials}")
    # single-run modes (dsd, dssd, baseline, local_baseline, all) 不加额外参数

    # 组装目录名
    dir_parts = [mode, model_short, engine]
    if param_parts:
        dir_parts.extend(param_parts)
    dir_parts.append(timestamp)
    dir_name = "_".join(dir_parts)

    experiment_dir = os.path.abspath(os.path.join(_PROJECT_ROOT, "output", dir_name))
    os.makedirs(experiment_dir, exist_ok=True)

    # 更新 csv_path 指向实验目录
    args.csv_path = os.path.join(experiment_dir, "results.csv")

    # 保存实验目录路径到 args, 方便后续使用
    args.experiment_dir = experiment_dir

    print(f"\n{'━'*60}")
    print(f"  📁 实验输出目录: {experiment_dir}")
    print(f"{'━'*60}")

    # 保存实验配置到 config.txt, 方便事后查阅
    config_path = os.path.join(experiment_dir, "config.txt")
    with open(config_path, "w") as f:
        f.write(f"Experiment: {dir_name}\n")
        f.write(f"Timestamp:  {timestamp}\n")
        f.write(f"{'─'*50}\n")
        for k, v in sorted(vars(args).items()):
            if k in ("experiment_dir",):
                continue
            f.write(f"  {k}: {v}\n")

    return experiment_dir


def _auto_visualize(args, mode="auto"):
    """
    实验结束后自动生成可视化图表。
    CSV 数据和图表都保存在实验目录下。
    """
    experiment_dir = getattr(args, 'experiment_dir', None)
    if experiment_dir is None:
        data_dir = os.path.dirname(args.csv_path) or os.path.join(_PROJECT_ROOT, "output")
        output_dir = os.path.join(_PROJECT_ROOT, "figures")
    else:
        data_dir = experiment_dir
        output_dir = os.path.join(experiment_dir, "figures")

    print(f"\n{'─'*60}")
    print(f"  📊 自动生成可视化图表 (mode={mode})")
    print(f"     输出目录: {output_dir}")
    print(f"{'─'*60}")
    try:
        success = run_visualization(
            data_dir=data_dir,
            output_dir=output_dir,
            mode=mode,
        )
        if not success:
            print("  ⚠ 可视化跳过 (未找到对应数据文件)")
    except Exception as e:
        print(f"  ⚠ 可视化出错 (不影响实验数据): {e}")


def main():
    parser = argparse.ArgumentParser(description="UAV (Device) Client")
    parser.add_argument('--input', type=str,
                        default="Alan Turing theorized that computers would one day become ")
    parser.add_argument('--draft_model_name', type=str,
                        default="/Users/myrick/modelHub/Qwen3-1.7B")
    parser.add_argument('--device', type=str, default="auto",
                        help="Device: 'auto' (multi-GPU auto-split if >1 GPU, else cuda>mps>cpu), "
                             "'cuda:0' (single GPU), 'mps', 'cpu'")
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help="Comma-separated GPU IDs for multi-GPU, e.g. '0,1,2,3'. "
                             "Only used when --device=auto. Default: use all available GPUs.")
    parser.add_argument('--framework', type=str, default="auto",
                        choices=["auto", "mlx", "pytorch"],
                        help="Inference framework: 'auto' (Apple Silicon→MLX, CUDA→PyTorch), "
                             "'mlx' (force MLX), 'pytorch' (force PyTorch)")
    parser.add_argument('--engine', type=str, default="auto",
                        choices=["auto", "vllm", "pytorch"],
                        help="Inference engine: 'auto' (NVIDIA GPU + vLLM available → vLLM, else PyTorch), "
                             "'vllm' (force vLLM), 'pytorch' (force PyTorch HuggingFace)")
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
                                 "all", "benchmark", "kv_benchmark",
                                 "token_energy", "token_energy_batch",
                                 "token_energy_stream"],
                        help="Which mode to run: dsd, dssd, baseline (remote LLM), "
                             "local_baseline (local SLM), all (single run), "
                             "benchmark (multi-prompt multi-trial), "
                             "kv_benchmark (KV cache progressive benchmark), "
                             "token_energy (per-token energy, sequential), "
                             "token_energy_batch (per-token energy, concurrent batch), "
                             "or token_energy_stream (per-token energy, rate-based streaming)")
    parser.add_argument('--csv_path', type=str,
                        default=os.path.join(_PROJECT_ROOT, "output",
                                             "results_real_network.csv"),
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
    # ---------- KV Cache Benchmark 参数 ----------
    parser.add_argument('--kv_lengths', type=str, default="32,64,128,256,512,1024",
                        help="Comma-separated KV cache lengths for kv_benchmark mode "
                             "(default: '32,64,128,256,512,1024')")
    parser.add_argument('--gen_tokens', type=int, default=64,
                        help="Fixed number of tokens to generate per run in kv_benchmark "
                             "(default: 64)")
    parser.add_argument('--interactive', action='store_true',
                        help="Enable interactive method selection menu")
    # ---------- Token Energy Benchmark 参数 ----------
    parser.add_argument('--token_samples', type=int, default=20,
                        help="Number of prompt samples for token_energy mode (default: 20)")
    parser.add_argument('--token_max_tokens', type=int, default=128,
                        help="Max tokens to generate per sample in token_energy mode (default: 128)")
    parser.add_argument('--batch_repeats', type=int, default=1,
                        help="Number of repeat rounds for token_energy_batch/stream mode. "
                             "Each round runs a full experiment, then results are averaged. (default: 1)")
    parser.add_argument('--req_rate', type=float, default=10.0,
                        help="For token_energy_stream mode: request injection rate in requests/min. "
                             "E.g. 10 = one request every 6 seconds. (default: 10.0)")
    parser.add_argument('--duration', type=int, default=600,
                        help="For token_energy_stream mode: experiment duration in seconds. "
                             "After this time, experiment ends immediately. (default: 600 = 10 min)")
    parser.add_argument('--warmup', type=int, default=0,
                        help="For token_energy_stream mode: number of warmup requests to "
                             "pre-inject before the timed experiment starts. These requests "
                             "complete prefill first, ensuring GPU is at steady-state "
                             "concurrency from the start. (default: 0 = no warmup)")
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

    # ==================== 交互式方法选择 ====================
    if args.interactive or args.mode in ("benchmark", "kv_benchmark"):
        args = _interactive_method_selection(args)

    # 根据设备自动选择框架和引擎，加载小模型
    uav_node, tokenizer = create_draft_node(
        model_name=args.draft_model_name,
        device_str=args.device,
        framework=args.framework,
        args=args,
        gpu_ids=getattr(args, 'gpu_ids', None),
        engine=getattr(args, 'engine', 'auto'),
    )

    # ==================== 创建实验输出目录 ====================
    experiment_dir = _make_experiment_dir(args)

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
    need_bs = args.mode in ("dssd", "dsd", "baseline", "all", "benchmark",
                             "kv_benchmark")
    # token_energy, token_energy_batch 和 local_baseline 不需要 BS 连接

    if need_bs:
        # benchmark / kv_benchmark 模式下如果只跑 local_baseline 则不需要连接
        if args.mode in ("benchmark", "kv_benchmark") and args.bench_modes:
            bench_mode_list = [m.strip() for m in args.bench_modes.split(",")]
            need_bs = any(m in ("dssd", "dsd", "baseline") for m in bench_mode_list)

    if need_bs:
        client = UAVClient(bs_host=args.bs_addr, bs_port=args.bs_port)
        client.connect()
    else:
        client = None

    try:
        # ==================== KV Cache Benchmark 模式 ====================
        if args.mode == "kv_benchmark":
            kv_lengths = [int(x.strip()) for x in args.kv_lengths.split(",")]

            bench_modes = None
            if args.bench_modes:
                bench_modes = [m.strip() for m in args.bench_modes.split(",")]

            all_raw, summaries = run_kv_cache_benchmark(
                uav_node=uav_node,
                client=client,
                tokenizer=tokenizer,
                args=args,
                kv_lengths=kv_lengths,
                gen_tokens=args.gen_tokens,
                num_trials=args.num_trials,
                modes=bench_modes,
                tc_config=shaper.get_config() if shaper else None,
            )

            if summaries:
                summary_csv = args.csv_path.replace(".csv", "_kvcache_summary.csv")
                save_results(summaries, summary_csv)

        # ==================== Token Energy Benchmark 模式 ====================
        elif args.mode == "token_energy":
            summary = run_token_energy_benchmark(
                uav_node=uav_node,
                tokenizer=tokenizer,
                args=args,
                num_samples=args.token_samples,
                max_tokens=args.token_max_tokens,
                seed=args.seed,
            )
            print(f"\n[Token Energy] 完成! "
                  f"共 {summary['num_sequences']} 条序列, "
                  f"平均 {summary['total_mean_mj_per_token']:.2f} mJ/token")
            # 自动可视化
            _auto_visualize(args, mode="sequential")

        # ==================== Batch Token Energy Benchmark 模式 ====================
        elif args.mode == "token_energy_batch":
            summary = run_token_energy_batch_benchmark(
                uav_node=uav_node,
                tokenizer=tokenizer,
                args=args,
                num_samples=args.token_samples,
                max_tokens=args.token_max_tokens,
                seed=args.seed,
                num_repeats=args.batch_repeats,
            )
            n_rounds = summary.get('num_repeats', 1)
            std_str = ""
            if n_rounds > 1:
                std_str = f" ± {summary.get('decode_std_mj_per_token', 0):.2f}"
            print(f"\n[Token Energy Batch] 完成! "
                  f"共 {summary['num_requests']} 个并发请求 × {n_rounds} 轮, "
                  f"{summary['total_steps']} steps/轮, "
                  f"平均 {summary['total_mean_mj_per_token']:.2f}{std_str} mJ/token")
            # 自动可视化
            _auto_visualize(args, mode="batch")

        # ==================== Stream Token Energy Benchmark 模式 ====================
        elif args.mode == "token_energy_stream":
            summary = run_token_energy_stream_benchmark(
                uav_node=uav_node,
                tokenizer=tokenizer,
                args=args,
                pool_size=args.token_samples,
                max_tokens=args.token_max_tokens,
                seed=args.seed,
                num_repeats=args.batch_repeats,
                req_rate=args.req_rate,
                duration=args.duration,
                warmup=args.warmup,
            )
            n_rounds = summary.get('num_repeats', 1)
            std_str = ""
            if n_rounds > 1:
                std_str = f" ± {summary.get('decode_std_mj_per_token', 0):.2f}"
            print(f"\n[Token Energy Stream] 完成! "
                  f"rate={summary['req_rate']:.1f} req/min, "
                  f"duration={summary['duration']}s, "
                  f"共注入 {summary['total_injected']} 个请求 × {n_rounds} 轮, "
                  f"平均 {summary['decode_mean_mj_per_token']:.2f}{std_str} mJ/token")
            # 自动可视化
            _auto_visualize(args, mode="stream")

        # ==================== Benchmark 模式 ====================
        elif args.mode == "benchmark":
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
