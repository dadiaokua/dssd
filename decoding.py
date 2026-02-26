"""
decoding.py - 解码主循环与 Baseline

包含:
  - generate_DSD():  经典分布式投机解码
  - generate_DSSD(): 分布式拆分投机解码
  - baseline_autoregressive():       远程大模型自回归
  - baseline_local_autoregressive(): 本地小模型自回归
  - run_benchmark():  多 prompt / 多轮 benchmark 并汇总统计
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
from energy_monitor import EnergyMonitor


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
    energy_mon = EnergyMonitor(device=uav_node.device, framework=uav_node.framework)
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
    energy_stats = energy_mon.stop()
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
    energy_mon = EnergyMonitor(device=uav_node.device, framework=uav_node.framework)
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
    energy_stats = energy_mon.stop()
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
    同时支持 PyTorch 和 MLX 后端，带能耗监控。
    """
    max_len = args.max_len
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p

    print(f"[Local AR] Using framework: {uav_node.framework}")
    print(f"[Local AR] Generating {max_len} tokens ...")

    # 能耗监控
    energy_mon = EnergyMonitor(device=uav_node.device, framework=uav_node.framework)
    energy_mon.start()

    t0 = time.time()

    if uav_node.framework == "mlx":
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

    else:
        x = input_ids.to(uav_node.device)

        with torch.no_grad():
            for i in tqdm(range(max_len), desc="local AR (PyTorch)"):
                logits = uav_node.model(x).logits
                next_tok = sample(logits[:, -1, :], temperature, top_k, top_p)
                x = torch.cat((x, next_tok), dim=1)

        output_ids = x.cpu()

    wall_time = time.time() - t0

    # 停止能耗监控
    energy_stats = energy_mon.stop()

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    throughput = max_len / wall_time

    print(f"\n{'='*60}")
    print(f"=== Baseline Autoregressive (Local SLM) ===")
    print(f"{'='*60}")
    print(f"Generated text: \033[93m{generated}\033[0m")
    print(f"Framework: {uav_node.framework}")
    print(f"Time: {wall_time:.2f}s")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(f"Generated tokens: {max_len}")
    # 本地自回归无网络通信
    net_energy = EnergyMonitor.estimate_network_energy(0, 0,
        net_type=getattr(args, "net_type", "wifi"))

    print(f"\n--- Energy (UAV local) ---")
    print(EnergyMonitor.format_report(energy_stats, max_len, net_energy))

    result = {
        "method": "baseline_local_ar",
        "generated": generated,
        "framework": uav_node.framework,
        "wall_time": wall_time,
        "throughput": throughput,
        "generated_tokens": max_len,
    }
    for k, v in energy_stats.items():
        result[f"energy_{k}"] = v
    for k, v in net_energy.items():
        result[f"energy_{k}"] = v
    result["energy_total_mj"] = round(energy_stats["est_energy_mj"], 1)
    return result


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
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    keys = list(result.keys())
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(result)


def run_benchmark(uav_node, client, tokenizer, args,
                  prompts: list[str] | None = None,
                  num_trials: int = 3,
                  modes: list[str] | None = None):
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
