"""
decoding.py - 解码主循环与 Baseline

包含:
  - generate_DSD():  经典分布式投机解码
  - generate_DSSD(): 分布式拆分投机解码
  - baseline_autoregressive():       远程大模型自回归
  - baseline_local_autoregressive(): 本地小模型自回归
  - save_results():  CSV 结果保存
"""

import csv
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from dssd_net import UAVClient
from dssd_utils import sample
from energy_monitor import EnergyMonitor


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
