"""
uav_client.py - UAV 端 (设备/客户端)
在边缘设备上运行，加载小模型做 draft，通过 TCP 调用 BS 端做 verify

用法（在 UAV/边缘设备上执行）:

  Linux + CUDA:
    python uav_client.py \
        --draft_model_name ./LLM/opt-125m \
        --device cuda:0 \
        --bs_addr 192.168.1.100 \
        --bs_port 50051 \
        --gamma 4 --max_len 80

  Mac (Apple Silicon, 自动使用 MPS 加速):
    python uav_client.py \
        --draft_model_name /Users/myrick/modelHub/hub/Qwen3-0.6B \
        --device auto \
        --bs_addr 192.168.1.100

  Mac (强制 CPU):
    python uav_client.py \
        --draft_model_name /Users/myrick/modelHub/hub/Qwen3-0.6B \
        --device cpu \
        --bs_addr 192.168.1.100

  同一台机器测试:
    python uav_client.py --bs_addr 127.0.0.1 --device auto

依赖: torch, transformers（不需要额外安装 grpc）
"""

import argparse
import time
import csv
import os
from collections import OrderedDict
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from dssd_net import UAVClient
from dssd_utils import sample, tensor_nbytes, compress_logits, resolve_device, get_device_info


class UAVDraftNode:
    """UAV 端小模型 draft 逻辑（本地执行，支持 CUDA / MPS / CPU）"""

    def __init__(self, model, device: torch.device, args):
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        self.args = args
        print(f"[UAV] Draft model loaded on: {get_device_info(device)}")

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
        # MPS 的 multinomial 可能有兼容问题，回退到 CPU
        if diff.device.type == "mps":
            xj_prime = torch.multinomial(diff.cpu(), 1).unsqueeze(0)
        else:
            xj_prime = torch.multinomial(diff, 1).unsqueeze(0)
        return xj_prime


# ============ DSD 主循环 ============

def generate_DSD(uav_node: UAVDraftNode, client: UAVClient,
                 input_ids: torch.Tensor, tokenizer, args):
    """DSD: 经典分布式投机解码（真实网络通信）"""
    input_ids = input_ids.to(uav_node.device)
    max_total_len = args.max_len + input_ids.shape[1]

    total_comm = total_slm = 0.0
    rounds = correct_nums = reject_nums = 0

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
            bs_time = response.get("bs_compute_time", 0)

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

    return {
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


# ============ DSSD 主循环 ============

def generate_DSSD(uav_node: UAVDraftNode, client: UAVClient,
                  input_ids: torch.Tensor, tokenizer, args):
    """DSSD: 分布式拆分投机解码（真实网络通信）"""
    input_ids = input_ids.to(uav_node.device)
    max_total_len = args.max_len + input_ids.shape[1]

    total_comm = total_slm = 0.0
    rounds = correct_nums = reject_nums = 0

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
            #    注意: DSSD 只发送 token + 概率值（不发完整 logits）
            t_rpc_start = time.time()
            response = client.call({
                "method": "verify_dssd",
                "x_draft": x_draft.cpu(),
                "q_values": q_values,        # list[float], 非常小！
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
                # 全部接受: BS 返回 x_{gamma+1}
                xj = response["xj"]
                new_prefix = torch.cat([x_draft, xj.to(uav_node.device)], dim=1)
                if new_prefix.shape[1] > max_total_len:
                    new_prefix = new_prefix[:, :max_total_len]
            else:
                # 拒绝: BS 返回 j + P_j 分布, UAV 本地 resample
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

    return {
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


# ============ Baseline: 远程大模型自回归 ============

def baseline_autoregressive(client: UAVClient, input_ids: torch.Tensor, tokenizer, args):
    """Baseline: 调用 BS 端大模型做纯自回归生成"""
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

    return {
        "method": "baseline_ar",
        "generated": generated,
        "bs_throughput": response["throughput"],
        "bs_time": response["time_cost"],
        "wall_time": wall_time,
    }


# ============ 结果记录 ============

def save_results(results_list: list, csv_path: str):
    """将所有实验结果保存到 CSV"""
    if not results_list:
        return
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results_list[0].keys())
        if write_header:
            writer.writeheader()
        for row in results_list:
            writer.writerow(row)
    print(f"\nResults saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="UAV (Device) Client")
    parser.add_argument('--input', type=str,
                        default="Alan Turing theorized that computers would one day become ")
    parser.add_argument('--draft_model_name', type=str, default="/Users/myrick/modelHub/hub/Qwen3-0.6B")
    parser.add_argument('--device', type=str, default="auto",
                        help="Device: 'auto' (cuda>mps>cpu), 'cuda:0', 'mps', 'cpu'")
    parser.add_argument('--bs_addr', type=str, default="127.0.0.1",
                        help="BS server IP address")
    parser.add_argument('--bs_port', type=int, default=50051,
                        help="BS server port")
    parser.add_argument('--max_len', type=int, default=80)
    parser.add_argument('--gamma', type=int, default=4)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--top_p', type=float, default=0)
    parser.add_argument('--mode', type=str, default="all",
                        choices=["dsd", "dssd", "baseline", "all"],
                        help="Which mode to run: dsd, dssd, baseline, or all")
    parser.add_argument('--csv_path', type=str, default="results_real_network.csv",
                        help="Path to save results CSV")
    args = parser.parse_args()

    # 解析设备
    device = resolve_device(args.device)

    # 加载小模型和 tokenizer
    print(f"[UAV Client] Loading draft model: {args.draft_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.draft_model_name)
    draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model_name)
    input_ids = tokenizer.encode(args.input, return_tensors='pt')

    # 创建 UAV draft 节点
    uav_node = UAVDraftNode(draft_model, device, args)

    # 连接 BS 端
    client = UAVClient(bs_host=args.bs_addr, bs_port=args.bs_port)
    client.connect()

    results = []
    try:
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

        if results:
            save_results(results, args.csv_path)

    finally:
        client.close()
        print("\n[UAV Client] Done.")


if __name__ == "__main__":
    main()
