"""
decoding.py - 解码主循环与 Baseline

包含:
  - generate_DSD():  经典分布式投机解码
  - generate_DSSD(): 分布式拆分投机解码
  - baseline_autoregressive():       远程大模型自回归
  - baseline_local_autoregressive(): 本地小模型自回归
  - run_benchmark():  多 prompt / 多轮 benchmark 并汇总统计
  - run_token_energy_benchmark(): 逐 token 能耗记录 benchmark
  - save_results():  CSV 结果保存
"""

import csv
import os
import time
import copy
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from dssd_net import UAVClient
from dssd_utils import sample
from energy_monitor import EnergyMonitor, TokenEnergyTracker


# ============ Benchmark Prompt 集 ============
# 覆盖不同领域/长度，保证实验的多样性和统计意义

BENCHMARK_PROMPTS = [
    # 科技 / 计算机
    "Alan Turing theorized that computers would one day become ",
    "The development of artificial intelligence has revolutionized ",
    "In the field of machine learning, transformer architectures have ",
    # 科学
    "According to Einstein's theory of relativity, ",
    "The process of photosynthesis in plants involves ",
    # 社会 / 历史
    "The Industrial Revolution fundamentally changed the way ",
    "Throughout human history, the invention of writing has ",
    # 工程 / 应用
    "Edge computing enables real-time data processing by ",
    "5G networks provide significantly higher bandwidth compared to ",
    # 推理 / 数学
    "If a train travels at 120 kilometers per hour for 3 hours, ",
]


# ============ DSD 主循环 ============

def generate_DSD(uav_node, client: UAVClient,
                 input_ids: torch.Tensor, tokenizer, args):
    """DSD: 经典分布式投机解码（真实网络通信）"""
    input_ids = input_ids.to(uav_node.device)
    max_total_len = args.max_len + input_ids.shape[1]

    total_comm = total_slm = 0.0
    rounds = correct_nums = reject_nums = 0

    # 能耗监控 + 重置网络流量统计
    energy_mon = EnergyMonitor(device=uav_node.device, framework=uav_node.framework,
                               model=uav_node.model)
    client.reset_stats()
    energy_mon.start()

    torch.manual_seed(args.seed)
    prefix = input_ids

    with tqdm(total=max_total_len, desc="DSD (real network)") as pbar:
        pbar.update(prefix.shape[1])
        initial_len = prefix.shape[1]
        start_time = time.time()

        while prefix.shape[1] < max_total_len:
            old_len = prefix.shape[1]
            rounds += 1

            # 1. UAV 端: draft（本地计算）
            t0 = time.time()
            x_draft, q_probs, up_bytes = uav_node.draft_step_DSD(prefix, args.gamma)
            slm_time = time.time() - t0
            total_slm += slm_time

            # 2. 通过真实 TCP 发送到 BS 端 verify
            t_rpc_start = time.time()
            response = client.call({
                "method": "verify_dsd",
                "x_draft": x_draft.cpu(),
                "q_probs": q_probs.cpu(),
                "gamma": args.gamma,
                "temperature": args.temperature,
            })
            rpc_time = time.time() - t_rpc_start
            total_comm += rpc_time

            # 3. 解析 BS 返回结果
            n = response["n"]
            t_corr = response["t_corr"]
            correct_nums += response["correct_num"]
            reject_nums += response["reject_num"]

            # 4. UAV 端更新 prefix
            prefix = torch.cat([
                x_draft[:, :n + 1],
                t_corr.to(uav_node.device)
            ], dim=1)

            new_len = prefix.shape[1]
            pbar.update(new_len - old_len)

    total_time = time.time() - start_time
    total_tokens = prefix.shape[1] - initial_len
    throughput = total_tokens / total_time if total_time > 0 else 0
    acceptance_rate = correct_nums / (rounds * args.gamma) if rounds > 0 else 0

    # 停止能耗监控 + 获取网络流量
    # avg_seq_len: decode 阶段上下文从 initial_len 增长到 initial_len + total_tokens
    avg_seq_len = initial_len + total_tokens // 2
    energy_stats = energy_mon.stop(tokens_generated=total_tokens,
                                   avg_seq_len=avg_seq_len)
    traffic = client.get_traffic_stats()
    net_energy = EnergyMonitor.estimate_network_energy(
        traffic["net_tx_bytes"], traffic["net_rx_bytes"],
        net_type=getattr(args, "net_type", "wifi"))

    generated = tokenizer.decode(prefix[0], skip_special_tokens=True)
    print(f"\n{'='*60}")
    print(f"=== DSD Results (Real Network) ===")
    print(f"{'='*60}")
    print(f"Generated text: \033[91m{generated}\033[0m")
    print(f"Throughput: \033[91m{throughput:.2f}\033[0m tokens/s")
    print(f"Acceptance rate: {acceptance_rate:.3f}")
    print(f"Total rounds: {rounds}")
    print(f"Accept/Reject: {correct_nums}/{reject_nums}")
    print(f"Total time: {total_time:.2f}s")
    print(f"  - SLM (draft) time:  {total_slm:.2f}s")
    print(f"  - RPC (comm+verify): {total_comm:.2f}s")
    print(f"  - Avg RPC latency:   {total_comm/rounds*1000:.1f}ms/round")
    print(f"\n--- Energy (UAV local) ---")
    print(EnergyMonitor.format_report(energy_stats, total_tokens, net_energy))

    result = {
        "method": "DSD",
        "generated": generated,
        "throughput": throughput,
        "total_time": total_time,
        "acceptance_rate": acceptance_rate,
        "rounds": rounds,
        "correct_nums": correct_nums,
        "reject_nums": reject_nums,
        "total_slm": total_slm,
        "total_comm": total_comm,
    }
    for k, v in energy_stats.items():
        result[f"energy_{k}"] = v
    for k, v in net_energy.items():
        result[f"energy_{k}"] = v
    result["energy_total_mj"] = round(
        energy_stats["est_energy_mj"] + net_energy["net_total_energy_mj"], 1)
    result["energy_compute_mj"] = round(energy_stats.get("compute_energy_mj", 0), 1)
    result["energy_memory_mj"] = round(energy_stats.get("memory_energy_mj", 0), 1)
    result["energy_idle_mj"] = round(energy_stats.get("idle_energy_mj", 0), 1)
    return result


# ============ DSSD 主循环 ============

def generate_DSSD(uav_node, client: UAVClient,
                  input_ids: torch.Tensor, tokenizer, args):
    """DSSD: 分布式拆分投机解码（真实网络通信）"""
    input_ids = input_ids.to(uav_node.device)
    max_total_len = args.max_len + input_ids.shape[1]

    total_comm = total_slm = 0.0
    rounds = correct_nums = reject_nums = 0

    # 能耗监控 + 重置网络流量统计
    energy_mon = EnergyMonitor(device=uav_node.device, framework=uav_node.framework,
                               model=uav_node.model)
    client.reset_stats()
    energy_mon.start()

    torch.manual_seed(args.seed)
    prefix = input_ids

    with tqdm(total=max_total_len, desc="DSSD (real network)") as pbar:
        pbar.update(prefix.shape[1])
        initial_len = prefix.shape[1]
        start_time = time.time()

        while prefix.shape[1] < max_total_len:
            old_len = prefix.shape[1]
            rounds += 1

            # 1. UAV 端: DSSD draft（本地计算）
            t0 = time.time()
            x_draft, q_values, q_probs, up_bytes = uav_node.draft_step_DSSD(prefix, args.gamma)
            slm_time = time.time() - t0
            total_slm += slm_time

            # 2. 通过真实 TCP 发送到 BS 端 verify
            t_rpc_start = time.time()
            response = client.call({
                "method": "verify_dssd",
                "x_draft": x_draft.cpu(),
                "q_values": q_values,
                "gamma": args.gamma,
                "temperature": args.temperature,
            })
            rpc_time = time.time() - t_rpc_start
            total_comm += rpc_time

            # 3. 解析结果并处理
            j = response["j"]
            flag = response["flag"]
            correct_nums += response["correct_num"]
            reject_nums += response["reject_num"]

            prefix_len = prefix.shape[1]

            if flag == 1:
                xj = response["xj"]
                new_prefix = torch.cat([x_draft, xj.to(uav_node.device)], dim=1)
                if new_prefix.shape[1] > max_total_len:
                    new_prefix = new_prefix[:, :max_total_len]
            else:
                pj = response["pj"]
                xj_prime = uav_node.resample_DSSD(j, pj, q_probs)
                x_draft[:, prefix_len + j - 1] = xj_prime.to(x_draft.device)
                new_prefix = torch.cat([prefix, x_draft[:, prefix_len:prefix_len + j]], dim=1)

            prefix = new_prefix
            new_len = prefix.shape[1]
            pbar.update(new_len - old_len)

    total_time = time.time() - start_time
    total_tokens = prefix.shape[1] - initial_len
    throughput = total_tokens / total_time if total_time > 0 else 0
    acceptance_rate = correct_nums / (rounds * args.gamma) if rounds > 0 else 0

    # 停止能耗监控 + 获取网络流量
    avg_seq_len = initial_len + total_tokens // 2
    energy_stats = energy_mon.stop(tokens_generated=total_tokens,
                                   avg_seq_len=avg_seq_len)
    traffic = client.get_traffic_stats()
    net_energy = EnergyMonitor.estimate_network_energy(
        traffic["net_tx_bytes"], traffic["net_rx_bytes"],
        net_type=getattr(args, "net_type", "wifi"))

    generated = tokenizer.decode(prefix[0], skip_special_tokens=True)
    print(f"\n{'='*60}")
    print(f"=== DSSD Results (Real Network) ===")
    print(f"{'='*60}")
    print(f"Generated text: \033[91m{generated}\033[0m")
    print(f"Throughput: \033[91m{throughput:.2f}\033[0m tokens/s")
    print(f"Acceptance rate: {acceptance_rate:.3f}")
    print(f"Total rounds: {rounds}")
    print(f"Accept/Reject: {correct_nums}/{reject_nums}")
    print(f"Total time: {total_time:.2f}s")
    print(f"  - SLM (draft) time:  {total_slm:.2f}s")
    print(f"  - RPC (comm+verify): {total_comm:.2f}s")
    print(f"  - Avg RPC latency:   {total_comm/rounds*1000:.1f}ms/round")
    print(f"\n--- Energy (UAV local) ---")
    print(EnergyMonitor.format_report(energy_stats, total_tokens, net_energy))

    result = {
        "method": "DSSD",
        "generated": generated,
        "throughput": throughput,
        "total_time": total_time,
        "acceptance_rate": acceptance_rate,
        "rounds": rounds,
        "correct_nums": correct_nums,
        "reject_nums": reject_nums,
        "total_slm": total_slm,
        "total_comm": total_comm,
    }
    for k, v in energy_stats.items():
        result[f"energy_{k}"] = v
    for k, v in net_energy.items():
        result[f"energy_{k}"] = v
    result["energy_total_mj"] = round(
        energy_stats["est_energy_mj"] + net_energy["net_total_energy_mj"], 1)
    result["energy_compute_mj"] = round(energy_stats.get("compute_energy_mj", 0), 1)
    result["energy_memory_mj"] = round(energy_stats.get("memory_energy_mj", 0), 1)
    result["energy_idle_mj"] = round(energy_stats.get("idle_energy_mj", 0), 1)
    return result


# ============ Baseline: 远程大模型自回归 ============

def baseline_autoregressive(client: UAVClient, input_ids: torch.Tensor, tokenizer, args):
    """Baseline: 调用 BS 端大模型做纯自回归生成"""
    client.reset_stats()
    t0 = time.time()
    response = client.call({
        "method": "autoregressive",
        "input_ids": input_ids.cpu(),
        "max_len": args.max_len,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
    })
    wall_time = time.time() - t0

    # 网络流量统计
    traffic = client.get_traffic_stats()
    net_energy = EnergyMonitor.estimate_network_energy(
        traffic["net_tx_bytes"], traffic["net_rx_bytes"],
        net_type=getattr(args, "net_type", "wifi"))

    output_ids = response["output_ids"]
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"\n{'='*60}")
    print(f"=== Baseline Autoregressive (Remote LLM) ===")
    print(f"{'='*60}")
    print(f"Generated text: \033[91m{generated}\033[0m")
    print(f"BS compute time: {response['time_cost']:.2f}s")
    print(f"BS throughput: {response['throughput']:.2f} tokens/s")
    print(f"Wall time (with network): {wall_time:.2f}s")
    print(f"Effective throughput: {args.max_len / wall_time:.2f} tokens/s")
    tx_kb = traffic["net_tx_bytes"] / 1024
    rx_kb = traffic["net_rx_bytes"] / 1024
    print(f"\n--- Network Traffic ---")
    print(f"  📤 TX: {tx_kb:.1f} KB  📥 RX: {rx_kb:.1f} KB  "
          f"Total: {(tx_kb+rx_kb):.1f} KB")
    print(f"  🔋 Network energy: {net_energy['net_total_energy_mj']:.1f} mJ")

    result = {
        "method": "baseline_remote_ar",
        "generated": generated,
        "bs_throughput": response["throughput"],
        "bs_time": response["time_cost"],
        "wall_time": wall_time,
    }
    for k, v in net_energy.items():
        result[f"energy_{k}"] = v
    return result


# ============ Baseline: 本地小模型自回归 ============

def baseline_local_autoregressive(uav_node, input_ids: torch.Tensor, tokenizer, args):
    """
    Baseline: 本地小模型纯自回归生成（不依赖 BS 端）
    支持 PyTorch / MLX / vLLM 后端，带能耗监控。
    """
    max_len = args.max_len
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p

    print(f"[Local AR] Using framework: {uav_node.framework}")
    print(f"[Local AR] Generating {max_len} tokens ...")

    # 能耗监控
    energy_mon = EnergyMonitor(device=uav_node.device, framework=uav_node.framework,
                               model=uav_node.model)
    energy_mon.start()

    t0 = time.time()

    if uav_node.framework == "vllm":
        # ---- vLLM 高性能自回归 ----
        output_ids, actual_gen = uav_node.generate_ar(
            input_ids, max_tokens=max_len,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )
        # vLLM 可能生成少于 max_len 的 token (遇到 EOS)
        actual_tokens = actual_gen

    elif uav_node.framework == "mlx":
        import mlx.core as mx

        x_np = input_ids.cpu().numpy()
        x_mx = mx.array(x_np)

        for i in tqdm(range(max_len), desc="local AR (MLX)"):
            logits_all = uav_node.model(x_mx)
            last_logits = logits_all[0, -1, :]
            tok_id, _ = uav_node._mlx_sample(last_logits)
            next_tok = mx.array([[tok_id]])
            x_mx = mx.concatenate([x_mx, next_tok], axis=1)

        output_ids = torch.from_numpy(np.array(x_mx.astype(mx.int64))).long()
        actual_tokens = max_len

    else:
        # ---- PyTorch 自回归 ----
        x = input_ids.to(uav_node.device)

        with torch.no_grad():
            for i in tqdm(range(max_len), desc="local AR (PyTorch)"):
                logits = uav_node.model(x).logits
                next_tok = sample(logits[:, -1, :], temperature, top_k, top_p)
                x = torch.cat((x, next_tok), dim=1)

        output_ids = x.cpu()
        actual_tokens = max_len

    wall_time = time.time() - t0

    # 停止能耗监控
    # avg_seq_len: 从 input_ids 长度增长到 input_ids + actual_tokens
    initial_len = input_ids.shape[1]
    avg_seq_len = initial_len + actual_tokens // 2
    energy_stats = energy_mon.stop(tokens_generated=actual_tokens,
                                   avg_seq_len=avg_seq_len)

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    throughput = actual_tokens / wall_time if wall_time > 0 else 0

    print(f"\n{'='*60}")
    print(f"=== Baseline Autoregressive (Local SLM) ===")
    print(f"{'='*60}")
    print(f"Generated text: \033[93m{generated}\033[0m")
    print(f"Framework: {uav_node.framework}")
    print(f"Time: {wall_time:.2f}s")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(f"Generated tokens: {actual_tokens}")
    # 本地自回归无网络通信
    net_energy = EnergyMonitor.estimate_network_energy(0, 0,
        net_type=getattr(args, "net_type", "wifi"))

    print(f"\n--- Energy (UAV local) ---")
    print(EnergyMonitor.format_report(energy_stats, actual_tokens, net_energy))

    result = {
        "method": "baseline_local_ar",
        "generated": generated,
        "framework": uav_node.framework,
        "wall_time": wall_time,
        "throughput": throughput,
        "generated_tokens": actual_tokens,
    }
    for k, v in energy_stats.items():
        result[f"energy_{k}"] = v
    for k, v in net_energy.items():
        result[f"energy_{k}"] = v
    result["energy_total_mj"] = round(energy_stats["est_energy_mj"], 1)
    result["energy_compute_mj"] = round(energy_stats.get("compute_energy_mj", 0), 1)
    result["energy_memory_mj"] = round(energy_stats.get("memory_energy_mj", 0), 1)
    result["energy_idle_mj"] = round(energy_stats.get("idle_energy_mj", 0), 1)
    return result


# ============ KV Cache 递进式 Benchmark ============

def _make_kv_prefill_ids(tokenizer, target_len: int) -> torch.Tensor:
    """
    构造一个长度约为 target_len 的 input_ids, 用于 KV cache 递进实验。

    策略: 用一段固定 prompt 重复填充到目标长度, 保证不同长度之间
    除了 KV cache 大小不同外, 生成行为基本一致。
    """
    base_text = (
        "The quick brown fox jumps over the lazy dog. "
        "In the field of machine learning, transformer architectures have "
        "revolutionized natural language processing. "
        "Edge computing enables real-time data processing by moving computation "
        "closer to the data source. "
    )
    base_ids = tokenizer.encode(base_text, add_special_tokens=False)
    if len(base_ids) == 0:
        base_ids = list(range(100, 200))  # fallback

    # 重复 base_ids 直到达到 target_len
    repeated = []
    while len(repeated) < target_len:
        repeated.extend(base_ids)
    repeated = repeated[:target_len]

    return torch.tensor([repeated], dtype=torch.long)


def run_kv_cache_benchmark(
    uav_node, client, tokenizer, args,
    kv_lengths: list[int] | None = None,
    gen_tokens: int = 64,
    num_trials: int = 1,
    modes: list[str] | None = None,
    tc_config: dict | None = None,
):
    """
    KV Cache 递进式 Benchmark:
    对每个 KV cache 长度 (即 prefill 长度), 运行选定的方法,
    观察 总能耗 / 计算能耗 / memory 能耗 随 KV cache 增长的变化。

    参数:
      uav_node:    DraftNode (PyTorch / MLX)
      client:      UAVClient (可以为 None, 此时只跑 local_baseline)
      tokenizer:   分词器
      args:        CLI 参数
      kv_lengths:  KV cache 长度列表 (prefill token 数)
                   默认: [32, 64, 128, 256, 512, 1024]
      gen_tokens:  每次生成的 token 数 (固定, 控制变量)
      num_trials:  每个长度重复次数
      modes:       要跑的方法列表
      tc_config:   网络限速配置

    返回:
      all_raw:   所有单次结果 [{...}, ...]
      summaries: 按 (method, kv_length) 聚合后的统计
    """
    if kv_lengths is None:
        kv_lengths = [32, 64, 128, 256, 512, 1024]
    if modes is None:
        if client is not None:
            modes = ["dssd", "dsd", "baseline", "local_baseline"]
        else:
            modes = ["local_baseline"]

    total_runs = len(kv_lengths) * num_trials * len(modes)
    print(f"\n{'#'*60}")
    print(f"# KV CACHE BENCHMARK")
    print(f"# KV lengths: {kv_lengths}")
    print(f"# gen_tokens (fixed): {gen_tokens}")
    print(f"# {len(kv_lengths)} lengths × {num_trials} trials × "
          f"{len(modes)} modes = {total_runs} runs")
    print(f"{'#'*60}\n")

    # 覆盖 args.max_len 为固定生成长度
    orig_max_len = args.max_len
    args.max_len = gen_tokens

    raw_csv = args.csv_path.replace(".csv", "_kvcache_raw.csv")

    all_raw: list[dict] = []
    by_group: dict[str, list[dict]] = defaultdict(list)  # key = "method|kv_len"

    for kv_len in kv_lengths:
        print(f"\n{'*'*60}")
        print(f"* KV Cache Length (prefill) = {kv_len} tokens")
        print(f"{'*'*60}")

        input_ids = _make_kv_prefill_ids(tokenizer, kv_len)
        print(f"  Input shape: {input_ids.shape}")

        for trial in range(num_trials):
            trial_args = copy.copy(args)
            trial_args.seed = args.seed + trial * 1000 + kv_len
            trial_args.max_len = gen_tokens

            banner = (f"[KV Bench] kv_len={kv_len} "
                      f"trial {trial+1}/{num_trials}")

            for mode in modes:
                print(f"\n{'='*60}")
                print(f"{banner}  mode={mode}")
                print(f"{'='*60}")

                try:
                    if mode == "dssd" and client is not None:
                        r = generate_DSSD(uav_node, client, input_ids,
                                          tokenizer, trial_args)
                    elif mode == "dsd" and client is not None:
                        r = generate_DSD(uav_node, client, input_ids,
                                         tokenizer, trial_args)
                    elif mode == "baseline" and client is not None:
                        r = baseline_autoregressive(client, input_ids,
                                                    tokenizer, trial_args)
                    elif mode == "local_baseline":
                        r = baseline_local_autoregressive(uav_node, input_ids,
                                                          tokenizer, trial_args)
                    else:
                        print(f"  ⚠ Skipping mode={mode} (no BS connection)")
                        continue
                except Exception as e:
                    print(f"  ❌ Error in {mode}: {e}")
                    import traceback; traceback.print_exc()
                    continue

                # 附加 KV cache 相关字段
                r["kv_cache_len"] = kv_len
                r["gen_tokens"] = gen_tokens
                r["trial"] = trial + 1
                if tc_config is not None:
                    r.update(tc_config)

                all_raw.append(r)
                group_key = f"{r['method']}|{kv_len}"
                by_group[group_key].append(r)

                # 每跑完一条立即写入 CSV
                _append_one_result(r, raw_csv)

    # 恢复 args.max_len
    args.max_len = orig_max_len

    print(f"\n✅ All {len(all_raw)} raw results saved to {raw_csv}")

    # ========== 汇总: 按 (method, kv_length) 聚合 ==========
    summaries: list[dict] = []
    print(f"\n{'#'*60}")
    print(f"# KV CACHE BENCHMARK SUMMARY")
    print(f"{'#'*60}")

    # 按 method 分组
    methods_seen = []
    for r in all_raw:
        if r["method"] not in methods_seen:
            methods_seen.append(r["method"])

    # 打印表头
    header = (f"{'Method':<20s} {'KV Len':>8s} "
              f"{'Total(mJ)':>10s} {'Compute(mJ)':>12s} "
              f"{'Memory(mJ)':>12s} {'Idle(mJ)':>10s} "
              f"{'Throughput':>12s}")
    print(f"\n{header}")
    print("-" * len(header))

    for method in methods_seen:
        for kv_len in kv_lengths:
            group_key = f"{method}|{kv_len}"
            group_results = by_group.get(group_key, [])
            if not group_results:
                continue

            agg = _aggregate_results(group_results)
            agg["method"] = method
            agg["kv_cache_len"] = kv_len
            agg["gen_tokens"] = gen_tokens
            summaries.append(agg)

            # 取 mean 值
            total_e = agg.get("energy_total_mj_mean",
                              agg.get("energy_est_energy_mj_mean", 0))
            comp_e = agg.get("energy_compute_mj_mean", 0)
            mem_e = agg.get("energy_memory_mj_mean", 0)
            idle_e = agg.get("energy_idle_mj_mean", 0)
            tp = agg.get("throughput_mean",
                         agg.get("energy_throughput_mean", 0))

            row = (f"{method:<20s} {kv_len:>8d} "
                   f"{total_e:>10.1f} {comp_e:>12.1f} "
                   f"{mem_e:>12.1f} {idle_e:>10.1f} "
                   f"{tp:>12.2f}")
            print(row)

        print()  # 方法之间空一行

    return all_raw, summaries


# ============ Token-Level Energy Benchmark ============

def run_token_energy_benchmark(
    uav_node, tokenizer, args,
    num_samples: int = 20,
    max_tokens: int = 128,
    seed: int = 42,
):
    """
    逐 token 能耗记录 Benchmark。

    从三个数据集随机采样 prompt, 对每个 prompt 逐 token 生成,
    使用 Zeus 硬件计数器记录每个 token 位置的能耗。
    最终按 token 位置取平均, 保存到 CSV 并返回汇总。

    参数:
      uav_node:    DraftNode (PyTorch 后端, 需要逐 token 控制)
      tokenizer:   分词器
      args:        CLI 参数 (需包含 csv_path)
      num_samples: 采样 prompt 数量
      max_tokens:  每个 prompt 生成的最大 token 数
      seed:        随机种子

    返回:
      summary: dict, TokenEnergyTracker.summarize() 的结果
    """
    from dataset_loader import load_prompts

    print(f"\n{'#'*60}")
    print(f"# TOKEN-LEVEL ENERGY BENCHMARK")
    print(f"# Samples: {num_samples}, Max tokens: {max_tokens}")
    print(f"# Seed: {seed}")
    print(f"{'#'*60}\n")

    # 加载 prompt
    prompts = load_prompts(n=num_samples, seed=seed,
                           min_length=10, max_length=2000)

    # 推断 GPU 索引 — 多卡模型需要监控所有 GPU
    gpu_indices = None
    if hasattr(args, 'gpu_ids') and args.gpu_ids:
        gpu_indices = [int(x) for x in args.gpu_ids.split(",")]
    elif uav_node.framework == "vllm":
        # vLLM TP 模式: 从 tensor_parallel_size 推断所有 GPU
        try:
            tp_size = uav_node._llm.model_config.tensor_parallel_size
            gpu_indices = list(range(tp_size))
        except Exception:
            # 退回: 用 pynvml 获取 GPU 数量
            try:
                import pynvml
                pynvml.nvmlInit()
                gpu_indices = list(range(pynvml.nvmlDeviceGetCount()))
            except Exception:
                gpu_indices = [0]
    elif hasattr(uav_node, 'model') and hasattr(uav_node.model, 'hf_device_map'):
        # PyTorch 多卡: 从 device_map 获取所有 GPU
        devs = set()
        for v in uav_node.model.hf_device_map.values():
            if isinstance(v, int):
                devs.add(v)
            elif isinstance(v, str) and v.startswith("cuda"):
                devs.add(int(v.split(":")[-1]))
        gpu_indices = sorted(devs) if devs else [0]
    elif uav_node.device is not None and uav_node.device.type == "cuda":
        gpu_indices = [uav_node.device.index or 0]
    print(f"[TokenEnergy] 监控 GPU: {gpu_indices}")

    # vLLM 0.8.1 的 step() 是同步的, 可以直接用 TokenEnergyTracker 精确测量
    tracker = TokenEnergyTracker(gpu_indices=gpu_indices)
    print(f"[TokenEnergy] 使用 TokenEnergyTracker (Zeus 逐 token 精确测量)")

    # 逐 sample 生成
    all_per_sample = []  # 每个 sample 的详细信息

    for si, item in enumerate(prompts):
        prompt_text = item["prompt"]
        source = item["source"]
        input_ids = tokenizer.encode(prompt_text, return_tensors='pt')
        input_ids = input_ids.to(uav_node.device)
        initial_len = input_ids.shape[1]

        print(f"\n[{si+1}/{num_samples}] source={source}, "
              f"prompt_len={initial_len} tokens, "
              f"preview: {prompt_text[:60]}...")

        tracker.new_sequence()

        if uav_node.framework == "vllm":
            # vLLM 0.8.1: step() 同步执行 forward pass, 可精确测量每个 token 能耗
            t0 = time.time()
            output_ids, generated_count = uav_node.generate_ar_stepwise(
                input_ids, max_tokens=max_tokens,
                tracker=tracker,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            wall_time = time.time() - t0
            throughput = generated_count / wall_time if wall_time > 0 else 0

        else:
            # PyTorch: 逐 token 生成, 使用 KV cache 加速, 精确测量每个 token 能耗
            generated_count = 0
            t0 = time.time()
            eos_id = getattr(tokenizer, 'eos_token_id', None)

            torch.manual_seed(args.seed + si)
            past_key_values = None
            next_input = input_ids.clone()

            with torch.no_grad():
                for pos in range(max_tokens):
                    tracker.begin_token(pos)

                    outputs = uav_node.model(
                        next_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    logits = outputs.logits
                    past_key_values = outputs.past_key_values

                    next_tok = sample(logits[:, -1, :],
                                      args.temperature, args.top_k, args.top_p)
                    # 下一轮只输入新 token (KV cache 已保存历史)
                    next_input = next_tok

                    tracker.end_token(pos)
                    generated_count += 1

                    # 进度
                    if (pos + 1) % 10 == 0 or pos == 0:
                        elapsed = time.time() - t0
                        speed = generated_count / elapsed if elapsed > 0 else 0
                        print(f"    token {pos+1}/{max_tokens} "
                              f"({speed:.1f} tok/s)", end="\r")

                    # 检查 EOS
                    if eos_id is not None and next_tok.item() == eos_id:
                        break

            wall_time = time.time() - t0
            throughput = generated_count / wall_time if wall_time > 0 else 0
            # 释放 KV cache 内存
            del past_key_values

        print(f"  Generated {generated_count} tokens in {wall_time:.2f}s "
              f"({throughput:.1f} tok/s)")

        # 记录每个 sample 的信息
        sample_info = {
            "sample_idx": si,
            "source": source,
            "prompt_len": initial_len,
            "generated_tokens": generated_count,
            "wall_time": wall_time,
            "throughput": throughput,
        }
        all_per_sample.append(sample_info)

    # 汇总
    summary = tracker.summarize()
    print(f"\n{'#'*60}")
    print(f"# TOKEN-LEVEL ENERGY SUMMARY")
    print(f"# Sequences: {summary['num_sequences']}")
    print(f"# Max position: {summary['max_position']}")
    print(f"# Mean energy/token: {summary['total_mean_mj_per_token']:.2f} mJ")
    print(f"{'#'*60}")

    # 保存逐位置 CSV
    csv_dir = os.path.dirname(args.csv_path) or "."
    os.makedirs(csv_dir, exist_ok=True)
    token_csv = os.path.join(csv_dir, "token_energy_per_position.csv")

    with open(token_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["position", "mean_energy_mj", "std_energy_mj",
                          "min_energy_mj", "max_energy_mj", "count"])
        for pos in range(len(summary["per_position_mean_mj"])):
            writer.writerow([
                pos,
                round(summary["per_position_mean_mj"][pos], 4),
                round(summary["per_position_std_mj"][pos], 4),
                round(summary["per_position_min_mj"][pos], 4),
                round(summary["per_position_max_mj"][pos], 4),
                summary["per_position_count"][pos],
            ])
    print(f"\n✅ Per-position energy saved to {token_csv}")

    # 保存每个 sample 的汇总
    sample_csv = os.path.join(csv_dir, "token_energy_per_sample.csv")
    with open(sample_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_per_sample[0].keys()))
        writer.writeheader()
        for row in all_per_sample:
            writer.writerow(row)
    print(f"✅ Per-sample summary saved to {sample_csv}")

    # 保存原始逐 token 数据 (每条 sequence 的每个 token)
    raw_csv = os.path.join(csv_dir, "token_energy_raw.csv")
    with open(raw_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence_idx", "position", "energy_mj"])
        for seq_idx, seq_data in enumerate(summary["all_energies"]):
            for pos, energy in seq_data:
                writer.writerow([seq_idx, pos, round(energy, 4)])
    print(f"✅ Raw token energy saved to {raw_csv}")

    return summary


# ============ Batch Token-Level Energy Benchmark ============

def _run_single_batch(uav_node, tokenizer, args, prompts, all_token_ids,
                       gpu_indices, max_tokens, round_idx, num_rounds):
    """
    执行单轮 batch token energy 测量。

    内部函数, 由 run_token_energy_batch_benchmark 调用。
    每轮创建独立的 TokenEnergyTracker, 确保数据隔离。

    返回:
        (round_result, tracker_raw, batch_results, step_info, wall_time)
        round_result: dict, 本轮的 per-position per-token energy (position → energy_mj)
        tracker_raw: list of (position, energy_mj), 原始 step 数据
        batch_results: list of dict, 每条请求的结果
        step_info: dict, prefill/decode 步数信息
        wall_time: float, 本轮耗时 (秒)
    """
    from energy_monitor import TokenEnergyTracker
    from collections import defaultdict

    num_samples = len(all_token_ids)

    print(f"\n{'='*60}")
    print(f"  ROUND {round_idx + 1} / {num_rounds}")
    print(f"{'='*60}")

    tracker = TokenEnergyTracker(gpu_indices=gpu_indices)

    t0 = time.time()
    batch_results, step_info = uav_node.generate_ar_stepwise_batch(
        prompts_token_ids=all_token_ids,
        max_tokens=max_tokens,
        tracker=tracker,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    wall_time = time.time() - t0

    total_generated = sum(r["generated_count"] for r in batch_results)
    throughput = total_generated / wall_time if wall_time > 0 else 0

    print(f"  [Round {round_idx + 1}] 生成 {total_generated} tokens in {wall_time:.1f}s "
          f"({throughput:.1f} tok/s)")

    # 提取 step 能耗
    tracker.finish()
    all_step_energies = tracker._all_sequences[0] if tracker._all_sequences else []

    prefill_energies = [(pos, e) for pos, e in all_step_energies if pos < 0]
    decode_energies = [(pos, e) for pos, e in all_step_energies if pos >= 0]

    gen_counts = [r["generated_count"] for r in batch_results]

    # 计算 per-position per-token energy
    pos_per_token = {}  # position → per_token_energy_mj
    for step_idx, (pos, step_energy) in enumerate(decode_energies):
        active = sum(1 for gc in gen_counts if gc > pos)
        if active == 0:
            active = 1
        pos_per_token[pos] = step_energy / active

    prefill_total = sum(e for _, e in prefill_energies)
    prefill_per_req = prefill_total / num_samples if num_samples > 0 else 0

    decode_vals = list(pos_per_token.values())
    decode_mean = sum(decode_vals) / len(decode_vals) if decode_vals else 0

    print(f"  [Round {round_idx + 1}] Prefill: {step_info['prefill_steps']} steps, "
          f"{prefill_total:.0f} mJ total ({prefill_per_req:.0f} mJ/req)")
    print(f"  [Round {round_idx + 1}] Decode: {step_info['decode_steps']} steps, "
          f"mean={decode_mean:.1f} mJ/token")

    round_result = {
        "pos_per_token": pos_per_token,
        "prefill_total_mj": prefill_total,
        "prefill_per_request_mj": prefill_per_req,
        "decode_mean_mj": decode_mean,
        "total_generated": total_generated,
        "wall_time": wall_time,
        "throughput": throughput,
        "gen_counts": gen_counts,
    }

    return round_result, all_step_energies, batch_results, step_info, wall_time


def run_token_energy_batch_benchmark(
    uav_node, tokenizer, args,
    num_samples: int = 30,
    max_tokens: int = 512,
    seed: int = 42,
    num_repeats: int = 1,
):
    """
    并发批量逐 position 能耗记录 Benchmark (支持多轮重复取平均)。

    将所有请求一次性提交给 vLLM, 确保它们在同一个 batch 中执行。
    由于所有请求同时开始, 每个 step() 中所有请求的 decode position 是对齐的:
      - Step 0: 所有请求做 prefill → 各产出 1 个 token (position 0)
      - Step N: 所有请求做 decode → 各产出 1 个 token (position N)

    每个 step 的 GPU 能耗除以该 step 的 active 请求数,
    得到每个 position 的平均单 token 能耗。

    当 num_repeats > 1 时, 使用不同的 prompt (不同 seed) 重复多轮,
    每轮都是完整的一个 batch, 等上一轮完全结束后再开始下一轮。
    最终将所有轮次的 per-position 能耗取平均。

    ★ 关键约束: 所有请求必须在同一个 batch 中,
      否则后续插入的请求会打乱 position 对齐。
      需确保 num_samples ≤ max_num_seqs (默认 256) 且
      总 prompt tokens ≤ max_num_batched_tokens。

    参数:
      uav_node:    VLLMDraftNode (必须是 vLLM 引擎)
      tokenizer:   分词器
      args:        CLI 参数 (需包含 csv_path)
      num_samples: 并发请求数量 (所有请求在同一 batch)
      max_tokens:  每个请求生成的最大 token 数
      seed:        随机种子 (每轮递增: seed, seed+1, seed+2, ...)
      num_repeats: 重复轮次数 (默认 1, 即只跑 1 轮)

    返回:
      summary: dict, 包含 per-position 能耗统计 (多轮平均)
    """
    from dataset_loader import load_prompts
    from energy_monitor import TokenEnergyTracker
    from collections import defaultdict
    import math
    import numpy as np

    assert uav_node.framework == "vllm", \
        "Batch token energy benchmark 仅支持 vLLM 引擎 (--engine vllm)"

    print(f"\n{'#'*60}")
    print(f"# BATCH TOKEN-LEVEL ENERGY BENCHMARK")
    print(f"# Concurrent requests: {num_samples}")
    print(f"# Max tokens per request: {max_tokens}")
    print(f"# Repeats: {num_repeats}")
    print(f"# Base seed: {seed}")
    print(f"{'#'*60}\n")

    # 推断 GPU 索引
    gpu_indices = None
    if hasattr(args, 'gpu_ids') and args.gpu_ids:
        gpu_indices = [int(x) for x in args.gpu_ids.split(",")]
    else:
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_indices = list(range(pynvml.nvmlDeviceGetCount()))
        except Exception:
            gpu_indices = [0]
    print(f"[BatchTokenEnergy] 监控 GPU: {gpu_indices}")

    csv_dir = os.path.dirname(args.csv_path) or "."
    os.makedirs(csv_dir, exist_ok=True)

    # ==================== 多轮执行 ====================
    all_round_results = []       # list of round_result dicts
    all_round_step_energies = [] # list of raw step energy lists
    all_round_batch_results = [] # list of batch_results
    all_round_step_infos = []    # list of step_info dicts
    all_round_prompts = []       # list of prompts

    total_wall_time = 0.0

    for r_idx in range(num_repeats):
        round_seed = seed + r_idx

        # 每轮加载不同的 prompt (不同 seed)
        prompts = load_prompts(n=num_samples, seed=round_seed,
                               min_length=10, max_length=2000)
        all_token_ids = [tokenizer.encode(item["prompt"]) for item in prompts]

        prompt_lens = [len(ids) for ids in all_token_ids]
        print(f"\n[Round {r_idx + 1}/{num_repeats}] seed={round_seed}, "
              f"{num_samples} prompts, "
              f"prompt lengths: min={min(prompt_lens)}, max={max(prompt_lens)}, "
              f"avg={sum(prompt_lens)/len(prompt_lens):.0f}")

        round_result, raw_steps, batch_results, step_info, wall_time = \
            _run_single_batch(
                uav_node, tokenizer, args, prompts, all_token_ids,
                gpu_indices, max_tokens, r_idx, num_repeats,
            )

        all_round_results.append(round_result)
        all_round_step_energies.append(raw_steps)
        all_round_batch_results.append(batch_results)
        all_round_step_infos.append(step_info)
        all_round_prompts.append(prompts)
        total_wall_time += wall_time

    # ==================== 跨轮次聚合 ====================
    print(f"\n{'='*60}")
    print(f"  AGGREGATING {num_repeats} ROUNDS")
    print(f"{'='*60}")

    # 收集所有轮次的 per-position per-token energy
    # 每轮每个 position 有一个值, 跨轮取平均/std
    pos_all_rounds = defaultdict(list)  # position → [energy_round1, energy_round2, ...]
    for rr in all_round_results:
        for pos, energy in rr["pos_per_token"].items():
            pos_all_rounds[pos].append(energy)

    max_pos = max(pos_all_rounds.keys()) if pos_all_rounds else 0

    # 聚合 prefill
    all_prefill_total = [rr["prefill_total_mj"] for rr in all_round_results]
    all_prefill_per_req = [rr["prefill_per_request_mj"] for rr in all_round_results]
    avg_prefill_total = np.mean(all_prefill_total)
    avg_prefill_per_req = np.mean(all_prefill_per_req)
    std_prefill_per_req = np.std(all_prefill_per_req, ddof=1) if num_repeats > 1 else 0.0

    # 聚合 throughput
    total_tokens_all = sum(rr["total_generated"] for rr in all_round_results)
    avg_throughput = total_tokens_all / total_wall_time if total_wall_time > 0 else 0

    # ---- 保存 prefill 能耗 (多轮汇总) ----
    prefill_csv = os.path.join(csv_dir, "token_energy_batch_prefill.csv")
    with open(prefill_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "prefill_total_mj", "prefill_per_request_mj"])
        for r_idx, rr in enumerate(all_round_results):
            writer.writerow([r_idx, round(rr["prefill_total_mj"], 4),
                             round(rr["prefill_per_request_mj"], 4)])
        # 最后一行写平均
        writer.writerow(["avg", round(avg_prefill_total, 4),
                         round(avg_prefill_per_req, 4)])
    print(f"✅ Prefill energy saved to {prefill_csv} "
          f"(avg total={avg_prefill_total:.1f} mJ, "
          f"avg per_req={avg_prefill_per_req:.1f} ± {std_prefill_per_req:.1f} mJ)")

    # ---- 保存每轮的 step 级原始数据 ----
    step_raw_csv = os.path.join(csv_dir, "token_energy_batch_step_raw.csv")
    with open(step_raw_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "decode_step", "position", "step_energy_mj",
                          "active_requests", "per_token_energy_mj"])
        for r_idx, (raw_steps, rr) in enumerate(
                zip(all_round_step_energies, all_round_results)):
            decode_energies = [(pos, e) for pos, e in raw_steps if pos >= 0]
            gen_counts = rr["gen_counts"]
            for step_idx, (pos, step_energy) in enumerate(decode_energies):
                active = sum(1 for gc in gen_counts if gc > pos)
                if active == 0:
                    active = 1
                per_token = step_energy / active
                writer.writerow([r_idx, step_idx, pos, round(step_energy, 4),
                                 active, round(per_token, 4)])
    print(f"✅ Step-level raw data saved to {step_raw_csv}")

    # ---- 保存 per-position CSV (多轮平均) ----
    token_csv = os.path.join(csv_dir, "token_energy_batch_per_position.csv")
    all_per_token_vals = []
    with open(token_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["position", "mean_energy_mj", "std_energy_mj",
                          "min_energy_mj", "max_energy_mj", "count",
                          "active_requests", "step_energy_mj",
                          "phase"])
        # Position -1: prefill (per-request 平均)
        writer.writerow([
            -1, round(avg_prefill_per_req, 4), round(std_prefill_per_req, 4),
            round(min(all_prefill_per_req), 4), round(max(all_prefill_per_req), 4),
            num_repeats, num_samples, round(avg_prefill_total, 4),
            "prefill",
        ])

        for p in range(max_pos + 1):
            vals = pos_all_rounds.get(p, [])
            if not vals:
                continue
            m = np.mean(vals)
            all_per_token_vals.append(m)
            std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            # active_requests: 用第一轮的 gen_counts 来估算 (各轮可能略有不同)
            active = sum(1 for gc in all_round_results[0]["gen_counts"] if gc > p)
            if active == 0:
                active = 1
            step_e = m * active
            writer.writerow([
                p, round(m, 4), round(std, 4),
                round(min(vals), 4), round(max(vals), 4),
                len(vals), active, round(step_e, 4),
                "decode",
            ])
    print(f"✅ Per-position energy saved to {token_csv}")

    # ---- 保存 per-sample summary (所有轮次) ----
    sample_csv = os.path.join(csv_dir, "token_energy_batch_per_sample.csv")
    with open(sample_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "sample_idx", "source", "prompt_len",
                          "generated_tokens", "request_id"])
        for r_idx, (batch_results, prompts) in enumerate(
                zip(all_round_batch_results, all_round_prompts)):
            for i, (result, item) in enumerate(zip(batch_results, prompts)):
                writer.writerow([
                    r_idx, i, item["source"], result["prompt_len"],
                    result["generated_count"], result["request_id"],
                ])
    print(f"✅ Per-sample summary saved to {sample_csv}")

    # ---- 保存每轮汇总 ----
    rounds_csv = os.path.join(csv_dir, "token_energy_batch_rounds.csv")
    with open(rounds_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "seed", "total_generated", "wall_time_s",
                          "throughput_tok_s", "decode_mean_mj_per_token",
                          "prefill_total_mj", "prefill_per_request_mj"])
        for r_idx, rr in enumerate(all_round_results):
            writer.writerow([
                r_idx, seed + r_idx, rr["total_generated"],
                round(rr["wall_time"], 2), round(rr["throughput"], 2),
                round(rr["decode_mean_mj"], 2),
                round(rr["prefill_total_mj"], 2),
                round(rr["prefill_per_request_mj"], 2),
            ])
    print(f"✅ Per-round summary saved to {rounds_csv}")

    # ---- 汇总统计 ----
    decode_mean = np.mean(all_per_token_vals) if all_per_token_vals else 0
    decode_std = np.std([rr["decode_mean_mj"] for rr in all_round_results], ddof=1) \
        if num_repeats > 1 else 0.0

    # 用第一轮的 step_info 作为代表 (各轮的 prefill_steps 应一致)
    prefill_steps = all_round_step_infos[0]["prefill_steps"]
    decode_steps = all_round_step_infos[0]["decode_steps"]
    total_steps = all_round_step_infos[0]["total_steps"]

    summary = {
        "num_requests": num_samples,
        "num_repeats": num_repeats,
        "max_position": max_pos,
        "prefill_steps": prefill_steps,
        "decode_steps": decode_steps,
        "total_steps": total_steps,
        "total_generated_tokens": total_tokens_all,
        "wall_time_s": total_wall_time,
        "throughput_tok_s": avg_throughput,
        "total_mean_mj_per_token": decode_mean,
        "decode_mean_mj_per_token": decode_mean,
        "decode_std_mj_per_token": decode_std,
        "prefill_total_energy_mj": avg_prefill_total,
        "prefill_per_request_mj": avg_prefill_per_req,
        "per_position_mean_mj": [np.mean(pos_all_rounds.get(p, [0]))
                                  for p in range(max_pos + 1)],
    }

    print(f"\n{'#'*60}")
    print(f"# BATCH TOKEN-LEVEL ENERGY SUMMARY ({num_repeats} rounds)")
    print(f"# Concurrent requests: {num_samples}")
    print(f"# Repeats: {num_repeats}, Seeds: {seed}~{seed + num_repeats - 1}")
    print(f"# Prefill: {prefill_steps} steps, "
          f"{avg_prefill_total:.1f} mJ total, "
          f"{avg_prefill_per_req:.1f} ± {std_prefill_per_req:.1f} mJ/request")
    print(f"# Decode: {decode_steps} steps, {max_pos + 1} positions")
    print(f"# Max position: {max_pos}")
    print(f"# Total tokens generated (all rounds): {total_tokens_all}")
    print(f"# Total wall time: {total_wall_time:.1f}s")
    print(f"# Avg throughput: {avg_throughput:.1f} tok/s")
    if num_repeats > 1:
        round_means = [rr["decode_mean_mj"] for rr in all_round_results]
        print(f"# Per-round decode mean: "
              + ", ".join(f"{m:.1f}" for m in round_means) + " mJ/token")
    print(f"# Mean energy/token (decode, {num_repeats}-round avg): "
          f"{decode_mean:.2f} ± {decode_std:.2f} mJ")
    print(f"# Prefill energy/request ({num_repeats}-round avg): "
          f"{avg_prefill_per_req:.2f} mJ")
    print(f"{'#'*60}")

    return summary


# ============ Stream Token-Level Energy Benchmark ============

def run_token_energy_stream_benchmark(
    uav_node, tokenizer, args,
    pool_size: int = 50,
    max_tokens: int = 512,
    seed: int = 42,
    num_repeats: int = 1,
    req_rate: float = 10.0,
    duration: int = 600,
    warmup: int = 0,
):
    """
    流式逐 token 能耗记录 Benchmark: 基于速率的请求注入模型。

    使用 req_rate (请求/分钟) 和 duration (实验时长/秒) 控制请求注入:
      - 可选 warmup: 在计时前先注入 warmup 个请求完成 prefill, 确保
        实验开始时 GPU 已处于稳定的高并发状态
      - 按 req_rate 速率持续注入请求 (循环使用 prompt pool)
      - 达到 duration 后立即结束实验
      - 只记录 decode 阶段的能耗 (跳过 prefill)

    与 batch 模式的区别:
      - batch: 所有请求同时开始, position 完全对齐
      - stream: 请求按时间速率陆续到达, 更贴近真实服务场景

    不同 batch size 的对比通过 --stream_batch_sizes 参数实现:
      在 uav_client.py 中循环多轮, 每轮用不同的 max_num_seqs 重新创建
      vLLM 引擎, 从而控制 vLLM 调度器的 batch size 上限。
      本函数只负责单轮实验逻辑。

    参数:
      uav_node:    VLLMDraftNode (必须是 vLLM 引擎)
      tokenizer:   分词器
      args:        CLI 参数 (需包含 csv_path)
      pool_size:   prompt pool 大小 (从 dataset 中加载)
      max_tokens:  每个请求生成的最大 token 数
      seed:        随机种子
      num_repeats: 重复轮次数 (默认 1)
      req_rate:    每分钟注入的请求数 (默认 10.0 req/min)
      duration:    每轮实验时长, 秒 (默认 600s = 10min)
      warmup:      预热请求数 (默认 0, 不预热)

    返回:
      summary: dict, 包含 per-position 能耗统计
    """
    from dataset_loader import load_prompts
    from energy_monitor import TokenEnergyTracker
    from collections import defaultdict

    assert uav_node.framework == "vllm", \
        "Stream token energy benchmark 仅支持 vLLM 引擎 (--engine vllm)"

    expected_requests = int(req_rate * duration / 60)

    print(f"\n{'#'*60}")
    print(f"# STREAM TOKEN-LEVEL ENERGY BENCHMARK")
    print(f"# Request rate: {req_rate:.1f} req/min")
    print(f"# Duration: {duration}s ({duration/60:.1f} min)")
    print(f"# Expected requests per round: ~{expected_requests}")
    print(f"# Warmup requests: {warmup}")
    print(f"# Energy recording: decode only (prefill skipped)")
    print(f"# Max tokens per request: {max_tokens}")
    print(f"# Prompt pool size: {pool_size}")
    print(f"# Repeats: {num_repeats}")
    print(f"# Base seed: {seed}")
    print(f"{'#'*60}\n")

    # 推断 GPU 索引
    gpu_indices = None
    if hasattr(args, 'gpu_ids') and args.gpu_ids:
        gpu_indices = [int(x) for x in args.gpu_ids.split(",")]
    else:
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_indices = list(range(pynvml.nvmlDeviceGetCount()))
        except Exception:
            gpu_indices = [0]
    print(f"[StreamTokenEnergy] 监控 GPU: {gpu_indices}")

    csv_dir = os.path.dirname(args.csv_path) or "."
    os.makedirs(csv_dir, exist_ok=True)

    # ==================== 多轮执行 ====================
    all_round_pos_energies = defaultdict(list)
    all_round_summaries = []
    all_round_results = []
    total_wall_time = 0.0

    for r_idx in range(num_repeats):
        round_seed = seed + r_idx

        # 加载 prompt pool (每轮不同 seed)
        prompts = load_prompts(n=pool_size, seed=round_seed,
                               min_length=10, max_length=2000)
        all_token_ids = [tokenizer.encode(item["prompt"]) for item in prompts]

        prompt_lens = [len(ids) for ids in all_token_ids]
        print(f"\n[Round {r_idx + 1}/{num_repeats}] seed={round_seed}, "
              f"pool={pool_size} prompts, "
              f"prompt lengths: min={min(prompt_lens)}, max={max(prompt_lens)}, "
              f"avg={sum(prompt_lens)/len(prompt_lens):.0f}")

        print(f"\n{'='*60}")
        print(f"  ROUND {r_idx + 1} / {num_repeats}  "
              f"(rate={req_rate:.1f} req/min, duration={duration}s)")
        print(f"{'='*60}")

        tracker = TokenEnergyTracker(gpu_indices=gpu_indices)

        t0 = time.time()
        results, step_records, stream_info = uav_node.generate_ar_stepwise_stream(
            prompts_token_ids=all_token_ids,
            max_tokens=max_tokens,
            tracker=tracker,
            req_rate=req_rate,
            duration=duration,
            warmup=warmup,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        wall_time = time.time() - t0
        total_wall_time += wall_time

        total_generated = stream_info["total_generated"]
        total_injected = stream_info["total_injected"]
        warmup_count = stream_info.get("warmup_count", 0)
        experiment_injected = stream_info.get("experiment_injected", total_injected)
        throughput = total_generated / wall_time if wall_time > 0 else 0

        # 聚合本轮的 per-position 能耗
        round_pos_energies = defaultdict(list)
        for result in results:
            for pos, energy in result["per_position_energies"]:
                round_pos_energies[pos].append(energy)

        round_pos_means = {}
        for pos in sorted(round_pos_energies.keys()):
            vals = round_pos_energies[pos]
            round_pos_means[pos] = np.mean(vals)

        decode_vals = list(round_pos_means.values())
        decode_mean = np.mean(decode_vals) if decode_vals else 0
        max_pos = max(round_pos_means.keys()) if round_pos_means else 0

        print(f"  [Round {r_idx + 1}] 注入 {total_injected} 请求 "
              f"(warmup={warmup_count}, 正式={experiment_injected}), "
              f"生成 {total_generated} tokens in {wall_time:.1f}s "
              f"({throughput:.1f} tok/s)")
        print(f"  [Round {r_idx + 1}] 实际速率: {stream_info['actual_req_rate']:.1f} req/min, "
              f"Max position: {max_pos}, "
              f"Mean decode energy: {decode_mean:.1f} mJ/token")

        # 存入全局聚合
        for pos, vals in round_pos_energies.items():
            all_round_pos_energies[pos].extend(vals)

        all_round_results.append((results, prompts, step_records, stream_info))

        round_summary = {
            "round_idx": r_idx,
            "seed": round_seed,
            "total_injected": total_injected,
            "total_generated": total_generated,
            "wall_time": wall_time,
            "throughput": throughput,
            "actual_req_rate": stream_info["actual_req_rate"],
            "decode_mean_mj": decode_mean,
            "max_position": max_pos,
            "num_steps": stream_info["total_steps"],
        }
        all_round_summaries.append(round_summary)

    # ==================== 跨轮次聚合 ====================
    print(f"\n{'='*60}")
    print(f"  AGGREGATING {num_repeats} ROUNDS")
    print(f"{'='*60}")

    final_max_pos = max(all_round_pos_energies.keys()) if all_round_pos_energies else 0

    total_tokens_all = sum(rs["total_generated"] for rs in all_round_summaries)
    total_injected_all = sum(rs["total_injected"] for rs in all_round_summaries)
    avg_throughput = total_tokens_all / total_wall_time if total_wall_time > 0 else 0

    # ---- 保存 per-position CSV ----
    token_csv = os.path.join(csv_dir, "token_energy_stream_per_position.csv")
    all_per_token_vals = []
    with open(token_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["position", "mean_energy_mj", "std_energy_mj",
                          "min_energy_mj", "max_energy_mj", "count", "phase"])

        for p in range(final_max_pos + 1):
            vals = all_round_pos_energies.get(p, [])
            if not vals:
                continue
            m = np.mean(vals)
            all_per_token_vals.append(m)
            std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            writer.writerow([
                p, round(m, 4), round(std, 4),
                round(min(vals), 4), round(max(vals), 4),
                len(vals), "decode",
            ])
    print(f"✅ Per-position energy saved to {token_csv}")

    # ---- 保存 per-sample summary ----
    sample_csv = os.path.join(csv_dir, "token_energy_stream_per_sample.csv")
    with open(sample_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "request_idx", "pool_idx", "source",
                          "prompt_len", "generated_tokens", "inject_time_s",
                          "is_warmup", "request_id"])
        for r_idx, (results, prompts, _, _) in enumerate(all_round_results):
            for result in results:
                pool_idx = result.get("pool_idx", result["idx"] % len(prompts))
                source = prompts[pool_idx].get("source", "unknown") \
                    if pool_idx < len(prompts) else "unknown"
                writer.writerow([
                    r_idx, result["idx"], pool_idx, source,
                    result["prompt_len"], result["generated_count"],
                    round(result.get("inject_time", 0), 2),
                    result.get("is_warmup", False),
                    result["request_id"],
                ])
    print(f"✅ Per-sample summary saved to {sample_csv}")

    # ---- 保存 step-level 记录 ----
    steps_csv = os.path.join(csv_dir, "token_energy_stream_steps.csv")
    with open(steps_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round", "step", "time_s", "step_energy_mj", "num_active",
            "num_prefill",
        ])
        for r_idx, (_, _, step_records, _) in enumerate(all_round_results):
            for row in step_records:
                writer.writerow([
                    r_idx,
                    row.get("step", 0),
                    round(row.get("time", 0.0), 4),
                    round(row.get("step_energy_mj", 0.0), 4),
                    row.get("num_active", 0),
                    row.get("num_prefill", 0),
                ])
    print(f"✅ Step-level records saved to {steps_csv}")

    # ---- 保存每轮汇总 ----
    rounds_csv = os.path.join(csv_dir, "token_energy_stream_rounds.csv")
    with open(rounds_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "seed", "total_injected", "total_generated",
                          "wall_time_s", "throughput_tok_s",
                          "actual_req_rate", "decode_mean_mj_per_token",
                          "max_position", "num_steps"])
        for rs in all_round_summaries:
            writer.writerow([
                rs["round_idx"], rs["seed"], rs["total_injected"],
                rs["total_generated"],
                round(rs["wall_time"], 2), round(rs["throughput"], 2),
                round(rs["actual_req_rate"], 2),
                round(rs["decode_mean_mj"], 2), rs["max_position"],
                rs["num_steps"],
            ])
    print(f"✅ Per-round summary saved to {rounds_csv}")

    # ---- 汇总统计 ----
    decode_mean = np.mean(all_per_token_vals) if all_per_token_vals else 0
    decode_std = np.std([rs["decode_mean_mj"] for rs in all_round_summaries], ddof=1) \
        if num_repeats > 1 else 0.0

    summary = {
        "req_rate": req_rate,
        "duration": duration,
        "pool_size": pool_size,
        "num_repeats": num_repeats,
        "warmup": warmup,
        "total_injected": total_injected_all,
        "max_position": final_max_pos,
        "total_generated_tokens": total_tokens_all,
        "wall_time_s": total_wall_time,
        "throughput_tok_s": avg_throughput,
        "decode_mean_mj_per_token": decode_mean,
        "decode_std_mj_per_token": decode_std,
    }

    print(f"\n{'#'*60}")
    print(f"# STREAM TOKEN-LEVEL ENERGY SUMMARY ({num_repeats} rounds)")
    print(f"# Request rate: {req_rate:.1f} req/min")
    print(f"# Duration: {duration}s per round")
    print(f"# Warmup: {warmup} requests (prefill completed before timing)")
    print(f"# Energy recording: decode only (prefill skipped)")
    print(f"# Total injected (all rounds): {total_injected_all}")
    print(f"# Repeats: {num_repeats}, Seeds: {seed}~{seed + num_repeats - 1}")
    print(f"# Max position: {final_max_pos}")
    print(f"# Total tokens generated (all rounds): {total_tokens_all}")
    print(f"# Total wall time: {total_wall_time:.1f}s")
    print(f"# Avg throughput: {avg_throughput:.1f} tok/s")
    if num_repeats > 1:
        round_means = [rs["decode_mean_mj"] for rs in all_round_summaries]
        print(f"# Per-round decode mean: "
              + ", ".join(f"{m:.1f}" for m in round_means) + " mJ/token")
    print(f"# Mean energy/token (decode, {num_repeats}-round avg): "
          f"{decode_mean:.2f} ± {decode_std:.2f} mJ")
    print(f"{'#'*60}")

    return summary


# ============ Benchmark 引擎 ============

# 需要取平均的数值字段 (前缀匹配)
_NUMERIC_KEYS = {
    "throughput", "total_time", "acceptance_rate", "rounds",
    "correct_nums", "reject_nums", "total_slm", "total_comm",
    "wall_time", "bs_time", "bs_throughput", "generated_tokens",
}


def _is_numeric_key(k: str) -> bool:
    """判断某个 key 是否应该做数值聚合"""
    if k in _NUMERIC_KEYS:
        return True
    if k.startswith("energy_"):
        return True
    return False


def _aggregate_results(results_list: list[dict]) -> dict:
    """
    对一组同方法的 result dict 做数值聚合：
    返回 mean / std / min / max，以及非数值字段取第一个值。
    """
    if not results_list:
        return {}

    agg = {}
    numeric_accum: dict[str, list[float]] = defaultdict(list)

    for r in results_list:
        for k, v in r.items():
            if _is_numeric_key(k):
                try:
                    numeric_accum[k].append(float(v))
                except (ValueError, TypeError):
                    pass

    # 非数值字段：取第一个
    first = results_list[0]
    for k, v in first.items():
        if not _is_numeric_key(k):
            agg[k] = v

    # 数值字段：mean / std / min / max（跳过空数组）
    for k, vals in numeric_accum.items():
        if not vals:
            continue
        arr = np.array(vals)
        agg[f"{k}_mean"] = round(float(arr.mean()), 4)
        agg[f"{k}_std"] = round(float(arr.std(ddof=0)), 4)
        agg[f"{k}_min"] = round(float(arr.min()), 4)
        agg[f"{k}_max"] = round(float(arr.max()), 4)

    agg["num_trials"] = len(results_list)
    return agg


def _append_one_result(result: dict, csv_path: str):
    """将单条结果追加写入 CSV（每跑完一条就写，防止崩溃丢数据）"""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

    if write_header:
        # 首条: 用当前 result 的 keys 作为 header
        keys = list(result.keys())
    else:
        # 后续: 读取已有 header, 并追加本条新增的 key (保持顺序)
        with open(csv_path, "r", newline="") as rf:
            reader = csv.reader(rf)
            keys = next(reader)
        for k in result.keys():
            if k not in keys:
                keys.append(k)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(result)


def run_benchmark(uav_node, client, tokenizer, args,
                  prompts: list[str] | None = None,
                  num_trials: int = 3,
                  modes: list[str] | None = None,
                  tc_config: dict | None = None):
    """
    多 prompt × 多轮 benchmark。

    参数:
      uav_node:   DraftNode (PyTorch / MLX)
      client:     UAVClient (可以为 None，此时只跑 local_baseline)
      tokenizer:  分词器
      args:       CLI 参数 (需包含 max_len, gamma, seed 等)
      prompts:    prompt 列表；None 则使用内置 BENCHMARK_PROMPTS
      num_trials: 每个 prompt 重复次数
      modes:      要跑的方法列表；None 则根据 client 是否存在自动决定
      tc_config:  网络限速配置 (来自 NetworkShaper.get_config())

    返回:
      all_raw:   所有单次结果 [{...}, ...]
      summaries: 按方法聚合后的统计 [{method, throughput_mean, ...}, ...]
    """
    if prompts is None:
        prompts = BENCHMARK_PROMPTS
    if modes is None:
        if client is not None:
            modes = ["dssd", "dsd", "baseline", "local_baseline"]
        else:
            modes = ["local_baseline"]

    total_runs = len(prompts) * num_trials * len(modes)
    print(f"\n{'#'*60}")
    print(f"# BENCHMARK: {len(prompts)} prompts × {num_trials} trials × "
          f"{len(modes)} modes = {total_runs} runs")
    print(f"# max_len = {args.max_len}")
    print(f"{'#'*60}\n")

    # 原始结果 CSV 路径（每跑完一条立即写入，防崩溃丢数据）
    raw_csv = args.csv_path.replace(".csv", "_raw.csv")

    all_raw: list[dict] = []
    # 按方法分组收集
    by_method: dict[str, list[dict]] = defaultdict(list)

    for pi, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        for trial in range(num_trials):
            # 每轮用不同 seed 保证多样性
            trial_args = copy.copy(args)
            trial_args.seed = args.seed + trial * 1000 + pi

            banner = (f"[Benchmark] prompt {pi+1}/{len(prompts)} "
                      f"trial {trial+1}/{num_trials}")

            for mode in modes:
                print(f"\n{'='*60}")
                print(f"{banner}  mode={mode}")
                print(f"{'='*60}")

                try:
                    if mode == "dssd" and client is not None:
                        r = generate_DSSD(uav_node, client, input_ids,
                                          tokenizer, trial_args)
                    elif mode == "dsd" and client is not None:
                        r = generate_DSD(uav_node, client, input_ids,
                                         tokenizer, trial_args)
                    elif mode == "baseline" and client is not None:
                        r = baseline_autoregressive(client, input_ids,
                                                    tokenizer, trial_args)
                    elif mode == "local_baseline":
                        r = baseline_local_autoregressive(uav_node, input_ids,
                                                          tokenizer, trial_args)
                    else:
                        print(f"  ⚠ Skipping mode={mode} (no BS connection)")
                        continue
                except Exception as e:
                    print(f"  ❌ Error in {mode}: {e}")
                    continue

                r["prompt"] = prompt[:60] + ("..." if len(prompt) > 60 else "")
                r["trial"] = trial + 1
                # 附加网络限速配置信息
                if tc_config is not None:
                    r.update(tc_config)
                all_raw.append(r)
                by_method[r["method"]].append(r)

                # ★ 每跑完一条立即写入 CSV，防止后续崩溃导致数据丢失
                _append_one_result(r, raw_csv)

    print(f"\n✅ All {len(all_raw)} raw results saved to {raw_csv}")

    # 汇总统计
    summaries: list[dict] = []
    print(f"\n{'#'*60}")
    print(f"# BENCHMARK SUMMARY ({len(all_raw)} successful runs)")
    print(f"{'#'*60}")

    for method_name, method_results in by_method.items():
        agg = _aggregate_results(method_results)
        agg["method"] = method_name + "_avg"
        summaries.append(agg)

        print(f"\n--- {method_name} ({len(method_results)} runs) ---")
        # 打印核心指标
        for metric in ["throughput", "total_time", "wall_time",
                        "acceptance_rate", "energy_total_mj",
                        "energy_compute_mj", "energy_memory_mj",
                        "energy_idle_mj",
                        "energy_est_energy_mj", "energy_net_total_energy_mj"]:
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"
            if mean_key in agg:
                print(f"  {metric:>30s}: "
                      f"{agg[mean_key]:>10.2f} ± {agg[std_key]:.2f}")

    return all_raw, summaries


# ============ 结果记录 ============

def save_results(results_list: list, csv_path: str):
    """将所有实验结果保存到 CSV（自动合并所有方法的字段）"""
    if not results_list:
        return
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    # 收集所有结果中出现过的字段，保持顺序
    all_keys: list[str] = []
    seen = set()
    for row in results_list:
        for k in row.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in results_list:
            full_row = {k: row.get(k, "") for k in all_keys}
            writer.writerow(full_row)
    print(f"\nResults saved to {csv_path}")
