# DSSD: Distributed Split Speculative Decoding for Efficient Edge Computing

> 面向边缘计算的分布式拆分投机解码框架 —— 在 UAV-BS 协作场景下加速 LLM 推理

## 项目简介

本项目实现了一个 **UAV（无人机/边缘设备）与 BS（基站/服务器）协作** 的 LLM 推理加速框架。核心思想是利用**投机解码（Speculative Decoding）** 技术，让边缘端的小模型快速生成候选 token，再由服务器端的大模型进行验证，从而在保证生成质量的同时显著降低延迟。

项目实现并对比了三种方法：

| 方法 | 说明 | 通信开销 |
|------|------|----------|
| **DSSD** (本项目核心) | 分布式拆分投机解码：UAV 只上传 token ID + 概率值 | 🟢 极低（几 KB/轮） |
| **DSD** (经典对照) | 分布式投机解码：UAV 上传完整 logits 分布 | 🟡 较高（几百 KB/轮） |
| **Baseline AR** | 纯自回归生成（本地小模型 / 远程大模型） | 🔴 无协作 / 全量远程 |

### DSSD 的核心优势

```
经典 DSD:  UAV → [token_ids + 完整logits (γ×V×4B)] → BS
本项目 DSSD: UAV → [token_ids + 概率值 (γ×4B)]     → BS
                    ↑ 通信量减少 ~V 倍 (V=词表大小, 通常 32K~150K)
```

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    UAV 端 (边缘设备)                         │
│                                                             │
│  uav_client.py ─── CLI 入口 + 流程编排                      │
│       │                                                     │
│       ├── draft_node.py                                     │
│       │     ├── PyTorchDraftNode (CUDA / MPS / CPU)         │
│       │     ├── MLXDraftNode     (Apple Silicon 原生加速)    │
│       │     └── create_draft_node() 工厂函数                 │
│       │                                                     │
│       ├── decoding.py                                       │
│       │     ├── generate_DSSD()                             │
│       │     ├── generate_DSD()                              │
│       │     ├── baseline_autoregressive()      (远程大模型)  │
│       │     ├── baseline_local_autoregressive() (本地小模型) │
│       │     └── save_results()                              │
│       │                                                     │
│       └── energy_monitor.py                                 │
│             └── EnergyMonitor (计算能耗 + 网络能耗)          │
│                                                             │
│  dssd_utils.py ─── 设备检测 / 采样 / tensor 工具            │
│  dssd_net.py ───── TCP 通信层 (UAVClient)                   │
└────────────────────────── TCP ──────────────────────────────┘
                             │
                             │  Socket + Pickle
                             │  (自动序列化 torch.Tensor)
                             │
┌────────────────────────────┴────────────────────────────────┐
│                    BS 端 (基站/服务器)                        │
│                                                             │
│  bs_server.py ─── 大模型加载 + 验证服务                      │
│       ├── BSVerifier                                        │
│       │     ├── _verify_dsd()                               │
│       │     ├── _verify_dssd()                              │
│       │     └── _autoregressive()                           │
│       └── load_target_model()                               │
│             ├── 单卡:     device="cuda:0"                   │
│             ├── 多卡切分: device="auto" (device_map)         │
│             └── CPU 卸载: --cpu_offload                     │
│                                                             │
│  dssd_net.py ───── TCP 通信层 (BSServer)                    │
│  dssd_utils.py ─── 共享工具                                  │
└─────────────────────────────────────────────────────────────┘
```

## 文件结构

```
DSSD-Efficient-Edge-Computing/
│
├── uav_client.py        # UAV 端入口 (CLI + main)          ~140 行
├── bs_server.py         # BS 端入口 (模型加载 + 验证)       ~350 行
│
├── draft_node.py        # Draft 节点 (PyTorch/MLX 双后端)   ~275 行
├── decoding.py          # 解码主循环 (DSD/DSSD/Baseline)    ~415 行
├── energy_monitor.py    # 能耗监控 (计算+网络)              ~480 行
├── dssd_utils.py        # 共享工具 (设备/采样/tensor)       ~120 行
├── dssd_net.py          # TCP 通信层 (零外部依赖)           ~165 行
│
├── main.py              # 原始单机模拟版本 (保留作参考)
├── speculative.py       # 原始投机解码工具函数 (保留作参考)
│
├── results/             # 历史实验结果
├── config/              # 实验配置脚本
└── README.md            # 本文件
```

## 快速开始

### 环境准备

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 基础依赖
pip install torch transformers tqdm

# BS 端多卡支持 (可选)
pip install accelerate

# Mac Apple Silicon 加速 (可选, 推荐)
pip install mlx-lm

# NVIDIA GPU 能耗监控 (可选)
pip install pynvml
```

### 启动 BS 端 (服务器)

```bash
# 单卡
python bs_server.py \
    --target_model_name Qwen/Qwen3-32B \
    --device cuda:0 \
    --port 50051 --verbose

# 多卡自动切分 (8 卡)
python bs_server.py \
    --target_model_name Qwen/Qwen3-32B \
    --device auto \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --dtype bf16 \
    --port 50051 --verbose

# 显存不够时启用 CPU offload
python bs_server.py \
    --target_model_name Qwen/Qwen3-32B \
    --device auto \
    --cpu_offload \
    --port 50051
```

### 启动 UAV 端 (边缘设备)

```bash
# Mac (自动检测 MLX)
python uav_client.py \
    --draft_model_name Qwen/Qwen3-0.6B \
    --device auto \
    --bs_addr 192.168.1.100 \
    --mode all

# Linux + CUDA
python uav_client.py \
    --draft_model_name Qwen/Qwen3-0.6B \
    --device cuda:0 \
    --bs_addr 192.168.1.100 \
    --mode all

# 仅本地 baseline (不需要 BS 服务器)
python uav_client.py \
    --draft_model_name Qwen/Qwen3-0.6B \
    --mode local_baseline

# 指定网络类型 (影响能耗估算)
python uav_client.py \
    --bs_addr 192.168.1.100 \
    --net_type lte \
    --mode all
```

### 同一台机器测试

```bash
# 终端 1: 启动 BS
python bs_server.py --target_model_name ./LLM/opt-1.3b --device cuda:0

# 终端 2: 启动 UAV (连接 localhost)
python uav_client.py --draft_model_name ./LLM/opt-125m --bs_addr 127.0.0.1
```

## 命令行参数

### uav_client.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--draft_model_name` | `Qwen3-0.6B` | 小模型路径 |
| `--device` | `auto` | 设备: `auto`, `cuda:0`, `mps`, `cpu` |
| `--framework` | `auto` | 框架: `auto`, `mlx`, `pytorch` |
| `--bs_addr` | `127.0.0.1` | BS 服务器 IP |
| `--bs_port` | `50051` | BS 服务器端口 |
| `--mode` | `all` | 运行模式: `dssd`, `dsd`, `baseline`, `local_baseline`, `all` |
| `--gamma` | `4` | 每轮 draft 的候选 token 数 |
| `--max_len` | `80` | 最大生成 token 数 |
| `--temperature` | `0.7` | 采样温度 |
| `--top_k` | `10` | Top-K 采样 |
| `--top_p` | `0` | Top-P (nucleus) 采样 |
| `--net_type` | `wifi` | 网络类型: `wifi`, `lte`, `eth` |
| `--csv_path` | `results_real_network.csv` | 结果输出路径 |

### bs_server.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--target_model_name` | `opt-1.3b` | 大模型路径 |
| `--device` | `auto` | 设备: `auto` (多卡), `cuda:0` (单卡), `cpu` |
| `--gpu_ids` | `None` | 指定 GPU, 如 `0,1,2` |
| `--cpu_offload` | `False` | 启用 CPU 卸载 |
| `--dtype` | `auto` | 精度: `auto`, `fp32`, `fp16`, `bf16` |
| `--port` | `50051` | 监听端口 |
| `--verbose` | `False` | 打印详细验证日志 |

## 框架自动选择

UAV 端会根据硬件自动选择最优推理框架：

| 硬件 | 自动选择 | 说明 |
|------|----------|------|
| Apple Silicon (M1/M2/M3/M4) | **MLX** | 原生 Metal 加速, 比 PyTorch MPS 快 ~45x |
| NVIDIA GPU | **PyTorch CUDA** | 标准 GPU 加速 |
| CPU | **PyTorch CPU** | 通用回退 |

可通过 `--framework mlx` 或 `--framework pytorch` 强制指定。

## 能耗监控

系统提供**进程级**能耗监控，不受其他应用影响：

```
[进程级指标 — 仅本进程，不受其他应用影响]
  ⏱  Wall time:          10.16s
  🖥  CPU time (process): 2.44s (user 2.07s + sys 0.37s)
  📈 CPU utilization:     24.0%
  💾 Peak memory (RSS):   1024.5 MB
  🔧 Compute type:        GPU 密集型 (含 GPU 能耗)
  ⚡ Device TDP:          44000 mW
  🔋 Compute energy:      312928 mJ (312.93 J)
  🔋 Compute avg power:   30800 mW

  [网络传输能耗 — WIFI]
  📤 TX (UAV→BS):        2.0 KB → 0.0 mJ
  📥 RX (BS→UAV):        50.0 KB → 0.5 mJ
  🔋 Network energy:      0.5 mJ

  [汇总]
  🔋 Total energy:        312929 mJ (312.93 J)
     (compute 312928 + network 0.5)
  🔋 Energy/token:        3911.6 mJ/token
```

### 监控策略

| 指标 | 来源 | 隔离性 |
|------|------|--------|
| CPU 时间 | `resource.getrusage(RUSAGE_SELF)` | ✅ 仅本进程 |
| 峰值内存 | `resource.getrusage` / `torch.cuda` | ✅ 仅本进程 |
| 计算能耗 | CPU 时间 × TDP / wall_time × TDP × load_factor | ✅ 进程级估算 |
| 网络能耗 | 实际传输字节数 × 网卡功耗 / 吞吐量 | ✅ 仅本进程流量 |

## 通信协议

UAV 与 BS 之间使用 **TCP + Pickle** 轻量通信，零外部依赖：

```
┌──────────────────────────────────────────┐
│  4 字节长度头  │  pickle 序列化数据       │
│  (big-endian)  │  (dict, 含 tensor 自动  │
│                │   CPU 序列化/反序列化)    │
└──────────────────────────────────────────┘
```

`UAVClient` 自动统计每次 RPC 的上行/下行字节数，用于网络能耗估算。

## DSSD vs DSD 通信对比

以 `gamma=4`, 词表大小 `V=151936` (Qwen3) 为例：

| 方法 | 上行内容 | 上行大小 | 下行内容 | 下行大小 |
|------|----------|----------|----------|----------|
| **DSSD** | token IDs + 概率值 | ~32 B | 验证结果 + resample 分布 | ~600 KB |
| **DSD** | token IDs + 完整 logits | ~2.4 MB | 验证结果 + 修正 token | ~8 B |

DSSD 将上行通信量从 **MB 级** 降低到 **字节级**，代价是下行需要传回 resample 用的概率分布。在上行带宽受限的边缘场景（如 UAV 通过 LTE 上行）中优势尤为明显。

## 原始模拟版本

`main.py` + `speculative.py` 是原始的单机模拟版本，使用 `time.sleep()` 模拟网络延迟。保留作为参考，实际实验请使用分布式版本 (`uav_client.py` + `bs_server.py`)。

## 依赖总结

| 包 | 必需 | 用途 |
|----|------|------|
| `torch` | ✅ | PyTorch 推理 |
| `transformers` | ✅ | 模型加载 (HuggingFace) |
| `tqdm` | ✅ | 进度条 |
| `accelerate` | BS 多卡时 | `device_map="auto"` 模型切分 |
| `mlx-lm` | Mac 可选 | Apple Silicon MLX 加速 |
| `pynvml` | 可选 | NVIDIA GPU 功耗监控 |

## License

MIT
