# DSSD: Distributed Split Speculative Decoding

> 面向边缘计算的分布式拆分投机解码框架 —— 在 UAV-BS 协作场景下加速 LLM 推理

## 项目简介

本项目实现了一个 **UAV（无人机/边缘设备）与 BS（基站/服务器）协作** 的 LLM 推理加速框架。核心思想是利用**投机解码（Speculative Decoding）** 技术，让边缘端的小模型快速生成候选 token，再由服务器端的大模型进行验证，从而在保证生成质量的同时显著降低延迟和能耗。

### 对比方法

| 方法 | 说明 | 通信开销 |
|------|------|----------|
| **DSSD** (本项目核心) | 分布式拆分投机解码：UAV 只上传 token ID + 概率值 | 极低（几 KB/轮） |
| **DSD** (经典对照) | 分布式投机解码：UAV 上传完整 logits 分布 | 较高（几百 KB/轮） |
| **Baseline AR** | 纯自回归生成（本地小模型 / 远程大模型） | 无协作 / 全量远程 |

### DSSD 核心优势

```
经典 DSD:  UAV → [token_ids + 完整logits (γ×V×4B)] → BS
本项目 DSSD: UAV → [token_ids + 概率值 (γ×4B)]     → BS
                    ↑ 通信量减少 ~V 倍 (V = 词表大小, 通常 32K~150K)
```

---

## 文件结构

```
dssd/
├── scripts/                         # 入口脚本 & 可视化
│   ├── uav_client.py                # UAV 端入口 (CLI + 实验编排)
│   ├── bs_server.py                 # BS 端入口 (大模型加载 + 验证服务)
│   ├── visualize_kvcache.py         # KV Cache 基准结果可视化
│   └── visualize_token_energy.py    # 逐 token 能耗结果可视化
│
├── src/                             # 核心源码模块
│   ├── draft_node.py                # Draft 节点 (PyTorch / vLLM / MLX)
│   ├── decoding.py                  # 解码主循环 (DSD / DSSD / Baseline)
│   ├── energy_monitor.py            # 能耗监控 (Zeus + 解析模型)
│   ├── dssd_utils.py                # 设备检测 / 采样 / tensor 工具
│   ├── dssd_net.py                  # TCP 通信层
│   ├── network_shaper.py           # OS 级网络限速 (macOS / Linux)
│   ├── dataset_loader.py            # 多数据集加载器
│   └── speculative.py               # 原始投机解码工具 (保留)
│
├── output/                          # CSV 实验结果
├── figures/                         # 可视化图表
├── dataset/                         # 数据集 (LongForm / python_code / WizardLM)
├── docs/                            # 详细文档
│   ├── startup_guide.md             # 实验启动指南 (所有模式 + 参数详解)
│   └── energy_model.md              # 能耗计算模型 (技术原理)
├── config/                          # 实验配置脚本
├── archive/                         # 历史代码备份
├── main.py                          # 原始单机模拟版本 (保留)
└── README.md
```

---

## 快速开始

### 1. 环境准备

```bash
python3 -m venv venv && source venv/bin/activate

pip install torch transformers accelerate tqdm
pip install zeus-ml          # 能耗监控 (推荐)
pip install vllm==0.8.1      # 高性能推理 (可选, NVIDIA GPU)
```

### 2. 本地自回归 (最简单，不需要 BS 服务器)

```bash
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-0.6B \
    --device auto \
    --mode local_baseline
```

### 3. 分布式推理 (需要 BS 服务器)

```bash
# 终端 1: 启动 BS 服务器
python scripts/bs_server.py \
    --target_model_name ~/model_hub/Qwen3-32B \
    --device auto --port 50051

# 终端 2: 启动 UAV 客户端
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-0.6B \
    --device auto \
    --bs_addr 127.0.0.1 \
    --mode all
```

### 4. 能耗实验

```bash
# KV Cache 递进测试
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-0.6B \
    --device auto \
    --mode kv_benchmark

# 逐 token 能耗记录
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-0.6B \
    --device auto \
    --mode token_energy

# 可视化
python scripts/visualize_kvcache.py
python scripts/visualize_token_energy.py
```

> **更多实验模式、完整参数说明、多 GPU 配置、网络限速等**，请参阅 [docs/startup_guide.md](docs/startup_guide.md)。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    UAV 端 (边缘设备)                         │
│                                                             │
│  scripts/uav_client.py ─── CLI 入口 + 流程编排               │
│       ├── src/draft_node.py                                 │
│       │     ├── PyTorchDraftNode (CUDA / MPS / CPU)         │
│       │     ├── VLLMDraftNode    (vLLM tensor parallelism)  │
│       │     └── MLXDraftNode     (Apple Silicon)            │
│       ├── src/decoding.py ─── 解码循环                       │
│       └── src/energy_monitor.py ─── 能耗监控                 │
│                                                             │
│  src/dssd_net.py ───── TCP 通信 (UAVClient)                 │
└────────────────────────── TCP ──────────────────────────────┘
                             │  Socket + Pickle
┌────────────────────────────┴────────────────────────────────┐
│                    BS 端 (基站/服务器)                        │
│                                                             │
│  scripts/bs_server.py ─── 大模型加载 + 验证服务              │
│       ├── BSVerifier (DSD / DSSD / AR 验证)                 │
│       └── load_target_model (单卡 / 多卡 / CPU offload)     │
│                                                             │
│  src/dssd_net.py ───── TCP 通信 (BSServer)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 通信协议

UAV 与 BS 之间使用 **TCP + Pickle** 轻量通信，零外部依赖：

```
┌──────────────────────────────────────────┐
│  4 字节长度头  │  pickle 序列化数据       │
│  (big-endian)  │  (dict, 含 tensor 自动  │
│                │   CPU 序列化/反序列化)    │
└──────────────────────────────────────────┘
```

### DSSD vs DSD 通信对比

以 `gamma=4`, 词表大小 `V=151936` (Qwen3) 为例：

| 方法 | 上行内容 | 上行大小 | 下行内容 | 下行大小 |
|------|----------|----------|----------|----------|
| **DSSD** | token IDs + 概率值 | ~32 B | 验证结果 + resample 分布 | ~600 KB |
| **DSD** | token IDs + 完整 logits | ~2.4 MB | 验证结果 + 修正 token | ~8 B |

---

## 详细文档

| 文档 | 内容 |
|------|------|
| [docs/startup_guide.md](docs/startup_guide.md) | **实验启动指南** — 所有实验模式详解、全部参数说明、多 GPU 配置、推理引擎选择、网络限速、典型实验配置、常见问题 |
| [docs/energy_model.md](docs/energy_model.md) | **能耗计算模型** — Zeus 集成原理、分析模型 (FLOPs/Bytes)、能耗拆分算法、GPU 能效常数、退化策略 |

---

## 依赖

| 包 | 必需 | 用途 |
|----|------|------|
| `torch` | ✅ | PyTorch 推理 |
| `transformers` | ✅ | HuggingFace 模型加载 |
| `tqdm` | ✅ | 进度条 |
| `accelerate` | 多卡时 | `device_map="auto"` 模型切分 |
| `zeus-ml` | 推荐 | GPU 能耗硬件计数器 (含 pynvml) |
| `vllm` | 可选 | 高性能推理引擎 (NVIDIA GPU) |
| `mlx-lm` | Mac 可选 | Apple Silicon MLX 加速 |
| `matplotlib` | 可视化 | 图表绘制 |
| `pandas` | 可视化 | 数据处理 |
| `pyarrow` | 数据集 | 读取 parquet 格式 |

## License

MIT
