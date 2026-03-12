import argparse
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from speculative import * 
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Tuple, List, Dict
import time
import torch.nn.functional as F
from tqdm import tqdm
import csv, time, os, json
from collections import OrderedDict

class UAVNode:
    """无人机节点 - 负责draft阶段的小模型推理"""
    
    def __init__(self, model, device, args):
        self.model = model.to(device)
        self.device = device
        self.args = args
        
    def draft_step(self, prefix, gamma):
        """
        UAV端执行draft步骤，生成gamma个候选token
        返回：
          x_draft: (1, prefix_len+γ) 
          q_step_logits_stack: (γ, V)
          dup_bytes: 传输字节数
        """
        x = prefix.to(self.device)
        q_stack = []
        
        with torch.no_grad():
            for _ in range(gamma):
                logits = self.model(x).logits  # (1, seq, V)
                q_stack.append(logits[0, -1].cpu())  # 存储最后一行 (V,)
                next_tok = sample(logits[:, -1, :],
                                self.args.temperature, 
                                self.args.top_k, 
                                self.args.top_p)
                x = torch.cat((x, next_tok), dim=1)
        
        q_step_logits = torch.stack(q_stack, dim=0)  # -> (γ, V)
        
        # 计算传输字节数
        raw_bytes = tensor_nbytes(q_step_logits)
        if self.args.use_dist_summary:
            comp_bytes = sum(tensor_nbytes(compress_logits(row)) for row in q_step_logits)
            dup_bytes = comp_bytes
        else:
            dup_bytes = raw_bytes
            
        return x, q_step_logits, dup_bytes
    
    def draft_DSSD(self, prefix: torch.Tensor, gamma: int) -> Tuple[torch.Tensor, List[float], torch.Tensor, int]:
        """
        DSSD Draft阶段：生成γ个推测token，记录概率值（上行传输）和完整分布（本地重新采样）
        返回：
          x_draft: 前缀+推测token（shape: (1, seq_len+gamma)）
          q_values: 每个推测token的概率值（list[float], 长度gamma）→ 上行传输
          q_probs: 每个推测token的完整分布（shape: (gamma, V)）→ 本地保存
          dup_bytes: 上行传输字节数（token+概率值）
        """
        x = prefix.to(self.device)
        q_probs = []  # 保存完整分布（本地用于重新采样）
        q_values = []  # 保存概率值（上行传输）

        with torch.no_grad():
            for _ in range(gamma):
                # 1. 小模型前向计算，得到当前分布Q_i(x)
                logits = self.model(x).logits  # (1, seq, V)
                q_dist = F.softmax(logits[0, -1], dim=-1).cpu()  # Q_i(x)：当前步骤的完整分布
                q_probs.append(q_dist)
                
                # 2. 采样下一个token x_i ~ Q_i(x)
                next_tok = sample(logits[:, -1, :], self.args.temperature, self.args.top_k, self.args.top_p)
                x = torch.cat((x, next_tok), dim=1)
                
                # 3. 记录x_i的概率值q_i = Q_i(x_i)（用于上行传输）
                tok_id = next_tok.item()
                q_values.append(q_dist[tok_id].item())  # q_i = P_Q(x_i)

        # 转换为张量（便于后续处理）
        q_probs = torch.stack(q_probs, dim=0)  # (gamma, V)
        
        # 计算上行传输字节数（token: int32；概率值: float32）
        token_bytes = x.numel() * 4  # 每个token占4字节
        prob_bytes = len(q_values) * 4  # 每个概率值占4字节
        dup_bytes = token_bytes + prob_bytes  # 总上行字节数（大幅减少）

        return x, q_values, q_probs, dup_bytes

    def resample_DSSD(self, j: int, pj: torch.Tensor, q_probs: torch.Tensor) -> torch.Tensor:
        """
        DSSD Resample阶段：设备用本地保存的Q_j分布和基站发送的P_j分布重新采样
        参数：
          j: 拒绝位置（1~gamma）
          pj: 基站返回的P_j分布（shape: (V,)）
          q_probs: 设备保存的Q分布（shape: (gamma, V)）
        返回：
          xj_prime: 重新采样的token（shape: (1, 1)）
        """
        q_j = q_probs[j-1]  # Q_j分布（对应拒绝位置j）
        diff = (pj - q_j).clamp(min=0)  # diff = max(0, P_j - Q_j)
        diff = diff / diff.sum()  # 归一化
        xj_prime = torch.multinomial(diff, 1).unsqueeze(0)  # 采样x_j'
        return xj_prime


class BSNode:
    """基站节点 - 负责verify阶段的大模型推理"""
    
    def __init__(self, model, device, args):
        self.model = model.to(device)
        self.device = device
        self.args = args
    
    def verify_step(self, x_draft, q_steps, gamma):
        """
        BS端执行verify步骤，验证UAV生成的候选token
        返回：
          n: 接受的token位置
          t_corr: 校正token
          correct_num: 接受的token数量  
          reject_num: 拒绝的token数量
        """
        correct_num = reject_num = 0
        prefix_len = x_draft.size(1) - gamma
        
        # 1) 大模型一次前向
        p_all = self.model(x_draft.to(self.device)).logits.cpu()  # (1, prefix_len+γ, V)
        p_slice = p_all[0, prefix_len-1 : prefix_len+gamma-1, :]  # 取 γ 行 -> (γ, V)

        # 2) softmax归一化
        p_probs = F.softmax(p_slice / self.args.temperature, dim=-1)  # (γ, V)
        q_probs = F.softmax(q_steps / self.args.temperature, dim=-1)  # (γ, V)

        # 3) accept/reject判断
        for i in range(gamma):
            tok_id = int(x_draft[0, prefix_len+i].item())
            if torch.rand(1).item() > (p_probs[i, tok_id] / q_probs[i, tok_id]):
                # 首次拒绝 → 回滚到 prefix_len+i-1
                n = prefix_len + i - 1
                diff = (p_probs[i] - q_probs[i]).clamp(min=0)
                diff = diff / diff.sum()
                t_corr = torch.multinomial(diff, 1).unsqueeze(0)  # (1,1)
                reject_num += 1
                return n, t_corr, correct_num, reject_num
            else:
                correct_num += 1
        
        # 全部通过：n = 最后位置
        n = prefix_len + gamma - 1
        prob_last = F.softmax(p_all[0, n] / self.args.temperature, dim=-1)
        t_corr = torch.multinomial(prob_last, 1).unsqueeze(0)  # shape (1,1)
        return n, t_corr, correct_num, reject_num
    
    def verify_DSSD(self, x_draft: torch.Tensor, q_values: List[float], gamma: int) -> Tuple[int, int, torch.Tensor, torch.Tensor, int, int]:
        """
        DSSD Verification阶段：用设备发送的概率值验证推测token（符合Algorithm 2）
        参数：
          x_draft: 设备生成的前缀+推测token（shape: (1, seq_len+gamma)）
          q_values: 设备发送的每个推测token的概率值（list[float], 长度gamma）
          gamma: 推测token数量
        返回：
          j: 处理到的位置（1~gamma+1）
          flag: 是否拒绝（0=拒绝，1=接受）
          pj: 拒绝位置的P_j分布（仅flag=0有效，shape: (V,)）
          xj: 接受时的新采样token（仅flag=1有效，shape: (1,1)）
          correct_num: 接受的token数量
          reject_num: 拒绝的token数量
        """
        correct_num = 0
        reject_num = 0
        prefix_len = x_draft.size(1) - gamma  # 前缀长度 = x_draft长度 - 推测token数

        # 1. 大模型并行计算P_1到P_{gamma+1}的分布（Algorithm 2的关键优化）
        with torch.no_grad():
            p_all = self.model(x_draft.to(self.device)).logits.cpu()  # (1, seq_len+gamma, V)
        p_logits = p_all[0, prefix_len-1 : prefix_len+gamma, :]  # (gamma+1, V) → P_1到P_{gamma+1}的logits
        p_probs = F.softmax(p_logits / self.args.temperature, dim=-1)  # (gamma+1, V) → 转换为概率分布

        # 2. 逐token验证（Algorithm 2的Accept/Reject逻辑）
        flag = 1  # 默认接受
        j = 1
        pj = None
        xj = None

        for i in range(gamma):
            current_j = i + 1  # 当前验证的位置（1~gamma）
            tok_id = int(x_draft[0, prefix_len + i].item())  # 推测token x_i的ID
            q_i = q_values[i]  # 设备发送的q_i = Q_i(x_i)（关键：仅用概率值）
            p_i = p_probs[i, tok_id].item()  # 基站计算的p_i = P_i(x_i)

            # 接受条件：r_j < min(1, p_i/q_i)（Algorithm 2的核心逻辑）
            if torch.rand(1).item() > min(1.0, p_i / q_i):
                # 拒绝：记录位置j和P_j分布（发送给设备重新采样）
                flag = 0
                j = current_j
                pj = p_probs[i]  # P_j分布（shape: (V,)）
                reject_num += 1
                break
            else:
                # 接受：继续验证下一个token
                correct_num += 1

        # 3. 处理全部接受的情况（j=gamma+1）
        if flag == 1:
            j = gamma + 1  # 处理到gamma+1位置
            xj = torch.multinomial(p_probs[gamma], 1).unsqueeze(0)  # 采样x_{gamma+1}~P_{gamma+1}

        return j, flag, pj, xj, correct_num, reject_num

def generate_DSD(uav_node, bs_node, input_ids, tokenizer, args):
    total_comm_raw = 0.0
    total_comm_comp = 0.0
    """
    分布式投机采样主循环，使用UAV和BS节点协作
    """
    input_ids = input_ids.to(uav_node.device)
    
    total_comm = total_slm = total_llm = 0.0
    rounds = correct_nums = reject_nums = 0
    total_dup_bytes = 0
    
    # === 新增：分别统计两种上传方案的通信耗时 ===
    total_comm_raw  = 0.0   # 直接上传完整 logits
    total_comm_comp = 0.0   # 上传 dist-summary
    
    # 准备参数
    max_total_len = args.max_len + input_ids.shape[1]
    rtt = args.rtt
    bandwidth = args.bandwidth

    torch.manual_seed(args.seed)
    prefix = input_ids

    # 主循环
    with tqdm(total=max_total_len, desc="distributed speculative sampling") as pbar:
        pbar.update(prefix.shape[1])
        initial_len = prefix.shape[1]
        dsp_start = time.time()
        
        while prefix.shape[1] < max_total_len:
            old_len = prefix.shape[1]
            rounds += 1
            
            # UAV端：执行draft步骤
            t0 = time.time()
            x_draft, q_probs, dup_bytes = uav_node.draft_step(prefix, args.gamma)               # 设备A
            total_dup_bytes += dup_bytes
            
            # 模拟上行传输延迟
            bw_Bps = args.bandwidth * 1e6 / 8  # Mbps → B/s
            t_up = tx_delay_bytes(dup_bytes, rtt, bw_Bps)
            
            # ---- 计算本轮两种方案的通信时延（不影响原有 sleep） ----
            raw_bytes  = tensor_nbytes(q_probs)
            comp_bytes = sum(tensor_nbytes(compress_logits(row)) for row in q_probs)
            comm_raw   = tx_delay_bytes(raw_bytes,  rtt, bw_Bps)
            comm_comp  = tx_delay_bytes(comp_bytes, rtt, bw_Bps)
            total_comm_raw  += comm_raw
            total_comm_comp += comm_comp
            
            total_slm += time.time() - t0
            total_comm += t_up
            time.sleep(t_up)

            # BS端：执行verify步骤
            t1 = time.time()
            n, t_corr, correct_num, reject_num = bs_node.verify_step(                           # 设备B
                x_draft, q_probs, args.gamma)
            correct_nums += correct_num
            reject_nums += reject_num
            total_llm += time.time() - t1
            
            # === 新增：统计下行通信时间（t_corr从BS返回到UAV）===
            down_bytes = tensor_nbytes(t_corr)          # 1 token → 4 bytes
            t_down = tx_delay_bytes(down_bytes, rtt, bw_Bps)
            total_comm_raw  += t_down                   # 两种方案下行相同
            total_comm_comp += t_down
            
            # UAV端：接收结果并更新prefix
            prefix = torch.cat([
                x_draft[:, : n+1],
                t_corr.to(uav_node.device)
            ], dim=1)

            # 更新进度条
            new_len = prefix.shape[1]
            pbar.update(new_len - old_len)
            
    dsp_time = time.time() - dsp_start
    total_tokens = prefix.shape[1] - initial_len
    dsp_throughput = total_tokens / dsp_time
    acceptance_rate = correct_nums / (rounds * args.gamma)

    generated = tokenizer.decode(prefix[0], skip_special_tokens=True)
    print("\n=== Distributed SP Results ===")
    print(f"Generated text: \033[91m{generated}\033[0m")
    print(f"Throughput: \033[91m{dsp_throughput:.2f}\033[0m tokens/s")
    print(f"Acceptance rate: {acceptance_rate:.3f}")
    print(f"Total rounds: {rounds}")
    print(f"Total accepted tokens: {correct_nums}")
    print(f"Total rejected tokens: {reject_nums}")
    print(f"Total proposed tokens: {rounds * args.gamma}")
    print(f"Accept/Reject ratio: {correct_nums}/{reject_nums} = {correct_nums/max(reject_nums,1):.2f}")
    print(f"Communication time (raw logits)   : {total_comm_raw:.3f} s")
    print(f"Communication time (dist summary) : {total_comm_comp:.3f} s") 
    print(f"Time saved with dist summary      : {(total_comm_raw-total_comm_comp):.3f} s")
    
    return generated, dsp_throughput, dsp_time, acceptance_rate, total_comm, total_dup_bytes, rounds, total_slm, total_llm, total_comm_raw, total_comm_comp    

def generate_DSSD(uav_node: UAVNode, bs_node: BSNode, input_ids: torch.Tensor, tokenizer: AutoTokenizer, args: argparse.Namespace) -> Tuple[str, float, float, float, float, int, int, float, float]:
    """
    DSSD主循环：设备与基站协作生成文本（符合Algorithm 2）
    返回：
      generated: 生成的文本
      throughput: 吞吐量（tokens/s）
      total_time: 总时间（秒）
      acceptance_rate: 接受率
      total_comm_delay: 总通信延迟（秒）
      total_dup_bytes: 上行传输总字节数（字节）
      rounds: 循环轮次
      total_slm_time: 设备小模型总时间（秒）
      total_llm_time: 基站大模型总时间（秒）
    """
    input_ids = input_ids.to(uav_node.device)
    max_total_len = args.max_len + input_ids.shape[1]  # 生成的总长度（输入+输出）
    bw_Bps = args.bandwidth * 1e6 / 8  # 带宽转换为字节/秒（B/s）

    # 初始化统计指标
    total_comm_delay = 0.0  # 总通信延迟（上行+下行）
    total_slm_time = 0.0    # 设备小模型（SLM）总时间
    total_llm_time = 0.0    # 基站大模型（LLM）总时间
    rounds = 0              # 循环轮次
    correct_num_total = 0   # 接受的token总数
    reject_num_total = 0    # 拒绝的token总数
    total_dup_bytes = 0     # 上行传输总字节数

    # 初始化前缀
    prefix = input_ids
    torch.manual_seed(args.seed)

    with tqdm(total=max_total_len, desc="DSSD: Distributed Split Speculative Decoding") as pbar:
        pbar.update(prefix.shape[1])  # 初始进度（输入长度）
        initial_len = prefix.shape[1]
        start_time = time.time()

        # 主循环：直到生成达到最大长度
        while prefix.shape[1] < max_total_len:
            old_len = prefix.shape[1]
            rounds += 1

            # 1. 设备端：Draft阶段（生成推测token+概率值）
            t_slm_start = time.time()
            # 调用修改后的draft_DSSD，获取token、概率值、完整分布、传输字节数
            x_draft, q_values, q_probs, dup_bytes = uav_node.draft_DSSD(prefix, args.gamma)
            total_dup_bytes += dup_bytes  # 统计上行传输总字节数（token+概率值）
            total_slm_time += time.time() - t_slm_start

            # 模拟上行传输延迟（RTT/2 + 序列化延迟）
            t_up = tx_delay_bytes(dup_bytes, args.rtt, bw_Bps)
            total_comm_delay += t_up
            time.sleep(t_up)  # 模拟传输时间

            # 2. 基站端：Verification阶段（用概率值验证）
            t_llm_start = time.time()
            # 调用修改后的verify_DSSD，传递概率值q_values
            j, flag, pj, xj, correct_num, reject_num = bs_node.verify_DSSD(x_draft, q_values, args.gamma)
            correct_num_total += correct_num
            reject_num_total += reject_num
            total_llm_time += time.time() - t_llm_start

            # 3. 设备端：处理基站反馈（Resample + Reset）
            prefix_len = prefix.shape[1]  # 当前前缀长度
            if flag == 1:
                # 情况1：全部接受（flag=1）→ 基站返回x_{gamma+1}
                new_prefix = torch.cat([x_draft, xj.to(uav_node.device)], dim=1)
                # 检查是否超出限制
                if new_prefix.shape[1] > max_total_len:
                    new_prefix = new_prefix[:, :max_total_len]
                # 下行传输内容：j=gamma+1（int） + xj（token）
                down_bytes = 4 + xj.numel() * 4  # j占4字节，xj占4字节
            else:
                # 情况2：拒绝（flag=0）→ 基站返回j（int） + P_j分布（tensor）
                # 设备用本地保存的q_probs和基站发送的pj重新采样x_j'
                xj_prime = uav_node.resample_DSSD(j, pj, q_probs)
                # 更新x_draft中的拒绝token（x_j→x_j'）
                x_draft[:, prefix_len + j - 1] = xj_prime.to(x_draft.device)
                # 新前缀：前缀 + 接受的token + 重新采样的token
                new_prefix = torch.cat([prefix, x_draft[:, prefix_len:prefix_len+j]], dim=1)
                # 下行传输内容：j（int） + P_j分布（tensor）
                down_bytes = 4 + tensor_nbytes(pj)  # j占4字节，pj占V*4字节（V为词表大小）

            # 模拟下行传输延迟
            t_down = tx_delay_bytes(down_bytes, args.rtt, bw_Bps)
            total_comm_delay += t_down
            time.sleep(t_down)  # 模拟传输时间

            # 更新前缀
            prefix = new_prefix
            # 更新进度条
            new_len = prefix.shape[1]
            pbar.update(new_len - old_len)

    # 4. 结果统计
    total_time = time.time() - start_time
    total_tokens = prefix.shape[1] - initial_len  # 生成的token总数（排除输入）
    throughput = total_tokens / total_time if total_time > 0 else 0.0
    acceptance_rate = correct_num_total / (rounds * args.gamma) if (rounds * args.gamma) > 0 else 0.0

    # 解码生成的文本
    generated = tokenizer.decode(prefix[0], skip_special_tokens=True)
    print("\n=== DSSD Results ===")
    print(f"Generated text: \033[91m{generated}\033[0m")
    print(f"Throughput: \033[91m{throughput:.2f}\033[0m tokens/s")
    print(f"Acceptance rate: {acceptance_rate:.3f}")
    print(f"Total rounds: {rounds}")
    print(f"Total accepted tokens: {correct_num_total}")
    print(f"Total rejected tokens: {reject_num_total}")
    print(f"Total proposed tokens: {rounds * args.gamma}")
    print(f"Accept/Reject ratio: {correct_num_total}/{reject_num_total} = {correct_num_total/max(reject_num_total,1):.2f}")
    print(f"Total communication delay: {total_comm_delay:.2f}s")
    print(f"Total SLM (device) time: {total_slm_time:.2f}s")
    print(f"Total LLM (BS) time: {total_llm_time:.2f}s")

    return generated, throughput, total_time, acceptance_rate, total_comm_delay, total_dup_bytes, rounds, total_slm_time, total_llm_time
    

def parse_arguments():
    parser = argparse.ArgumentParser(description='args')

    parser.add_argument('--input', type=str, default="Alan Turing theorized that computers would one day become ")
    parser.add_argument('--draft_model_name', type=str, default="./LLM/opt-125m")
    parser.add_argument('--target_model_name', type=str, default="./LLM/opt-1.3b") 
    parser.add_argument('--max_len', type=int, default=80) 
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--gamma', type=int, default=4)
    parser.add_argument('--rtt', type=float, default=0.02)
    parser.add_argument('--bandwidth', type=float, default=1000, help='Bandwidth in Mbps')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--top_p', type=float, default=0)
    parser.add_argument('--csv_path', type=str, default="results.csv")
    parser.add_argument('--device_1', type=str, default="cuda:6")
    parser.add_argument('--device_2', type=str, default="cuda:2")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--use_dist_summary', action='store_true', help='upload compressed distribution instead of raw logits')
    parser.add_argument('--no_cache', action='store_true', help='disable Δ-prompt cache (ablation)')
    return parser.parse_args()

args = parse_arguments()
recorder = Recorder(args.csv_path)

if __name__ == "__main__":
    args = parse_arguments()
    torch.cuda.empty_cache()  # 清理未使用的显存
    device_1 = torch.device(args.device_1) 
    device_2 = torch.device(args.device_2)
    tokenizer = AutoTokenizer.from_pretrained(args.draft_model_name)
    draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model_name)
    target_model = AutoModelForCausalLM.from_pretrained(args.target_model_name)
    input_ids = tokenizer.encode(args.input, return_tensors='pt')
    
    # 创建UAV和BS节点
    uav_node = UAVNode(draft_model, device_1, args)
    bs_node = BSNode(target_model, device_2, args)
    
    # 执行DSSD
    generated, dssd_throughput, total_time, acceptance_rate, total_comm, total_dup_bytes, rounds, total_slm, total_llm = \
        generate_DSSD(uav_node, bs_node, input_ids, tokenizer, args) 
    
    # 执行分布式投机采样
    generated, dsp_throughput, dsp_time, acceptance_rate, total_comm, total_dup_bytes, rounds, total_slm, total_llm, total_comm_raw, total_comm_comp = \
        generate_DSD(uav_node, bs_node, input_ids, tokenizer, args)
    
    torch.cuda.empty_cache() 
    _prefix, t_ar_llm, tp_ar_llm = normal_generate(target_model, tokenizer, input_ids, device_2, args)
    
    torch.cuda.empty_cache() 
    _prefix, t_ar_slm, tp_ar_slm = normal_generate(draft_model, tokenizer, input_ids, device_1, args)
    
    # 计算新增的eta表字段
    T_draft_ms = (total_slm / rounds) * 1000 if rounds > 0 else 0
    T_comm_ms = (total_comm / rounds) * 1000 if rounds > 0 else 0
    T_verify_ms = (total_llm / rounds) * 1000 if rounds > 0 else 0
    T_down_ms = (args.rtt / 2) * 1000  # 下行延时约为RTT/2
    T_total_ms = T_draft_ms + T_comm_ms + T_verify_ms + T_down_ms
    
    recorder.add_entry(
        model_s = args.draft_model_name.rstrip('/').split('/')[-1],
        model_l = args.target_model_name.rstrip('/').split('/')[-1],
        speedup_dssd = round(dssd_throughput/tp_ar_llm, 2),
        speedup_dsp = round(dsp_throughput/tp_ar_llm, 2),
        b_dssd = round(total_comm/max(t_ar_slm, 1e-4), 1),
        c_dssd = round(t_ar_slm/t_ar_llm, 1),
        accept_rate_dssd = round(acceptance_rate, 3),
        b_dsp = round(total_comm/max(t_ar_slm, 1e-4), 1),
        c_dsp = round(t_ar_slm/t_ar_llm, 1),
        accept_rate_dsp = round(acceptance_rate, 3),
        base_thr= round(tp_ar_llm, 2),
        slm_thr = round(tp_ar_slm, 2),
            gamma   = args.gamma,
            rtt_ms  = args.rtt*1e3,
            bw_Mbps = args.bandwidth,   
            prompt_len = args.max_len,
            t_ar_slm = round(t_ar_slm, 2),
            t_ar_llm = round(t_ar_llm, 2),
            dup_B    = round(total_dup_bytes / 1024, 1),   # KB
            # eta表新增字段
            temperature = args.temperature,
            top_k = args.top_k,
            top_p = args.top_p,
            avg_rounds = rounds,
            T_draft_ms = round(T_draft_ms, 1),
            T_comm_ms = round(T_comm_ms, 1),
            T_verify_ms = round(T_verify_ms, 1),
            T_down_ms = round(T_down_ms, 1),
            T_total_ms = round(T_total_ms, 1),
            # T_comm_raw_ms = round(total_comm_raw * 1000, 1),
            # T_comm_comp_ms = round(total_comm_comp * 1000, 1),
        )