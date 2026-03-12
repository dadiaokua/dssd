# 实验启动指南

> 本文档详细描述所有实验模式的启动方式、参数说明和典型配置。
>
> 能耗计算的技术原理（Zeus 集成、分析模型、能耗拆分算法）请参阅 [energy_model.md](energy_model.md)。

---

## 目录

- [环境准备](#环境准备)
- [项目结构](#项目结构)
- [UAV 客户端 (uav_client.py)](#uav-客户端)
  - [全部参数一览](#全部参数一览)
  - [实验模式详解](#实验模式详解)
    - [模式 1: 单次运行 (all)](#模式-1-单次运行-all)
    - [模式 2: 单方法运行 (dsd / dssd / baseline / local_baseline)](#模式-2-单方法运行)
    - [模式 3: 多轮基准测试 (benchmark)](#模式-3-多轮基准测试-benchmark)
    - [模式 4: KV Cache 递进测试 (kv_benchmark)](#模式-4-kv-cache-递进测试-kv_benchmark)
    - [模式 5: 逐 Token 能耗记录 (token_energy)](#模式-5-逐-token-能耗记录-token_energy)
    - [模式 6: 并发批量逐 Token 能耗记录 (token_energy_batch)](#模式-6-并发批量逐-token-能耗记录-token_energy_batch)
    - [模式 7: 流式逐 Token 能耗记录 (token_energy_stream)](#模式-7-流式逐-token-能耗记录-token_energy_stream)
  - [交互式方法选择](#交互式方法选择)
  - [网络限速 (Traffic Shaping)](#网络限速)
- [BS 服务端 (bs_server.py)](#bs-服务端)
  - [全部参数一览](#bs-全部参数一览)
  - [部署配置](#部署配置)
- [可视化脚本](#可视化脚本)
- [多 GPU 配置](#多-gpu-配置)
- [推理引擎选择](#推理引擎选择)
- [典型实验配置](#典型实验配置)

---

## 环境准备

```bash
cd /home/llm/github_project/dssd

# 创建虚拟环境 (首次)
python3 -m venv venv
source venv/bin/activate

# 基础依赖
pip install torch transformers accelerate tqdm numpy

# 能耗监控 (推荐)
pip install zeus-ml        # Zeus 能耗框架 (包含 pynvml)

# 高性能推理引擎 (可选, NVIDIA GPU)
pip install vllm==0.8.1    # vLLM (同步 step(), 支持逐 token 能耗测量)

# Apple Silicon 加速 (可选)
pip install mlx-lm          # MLX 框架

# 数据集读取
pip install pyarrow          # 读取 parquet 格式数据集

# 可视化
pip install matplotlib pandas
```

### 验证安装

```bash
source venv/bin/activate

# 验证 Zeus
python3 -c "from zeus.monitor.energy import ZeusMonitor; print('✅ Zeus OK')"

# 验证 vLLM
python3 -c "import vllm; print(f'✅ vLLM {vllm.__version__}')"

# 验证 GPU
python3 -c "import torch; print(f'✅ CUDA: {torch.cuda.device_count()} GPUs')"
```

---

## 项目结构

```
dssd/
├── scripts/                         # 入口脚本
│   ├── uav_client.py                # UAV 客户端 (主入口)
│   ├── bs_server.py                 # BS 服务端
│   ├── visualize_kvcache.py         # KV Cache 结果可视化
│   └── visualize_token_energy.py    # 逐 token 能耗可视化
├── src/                             # 核心模块
├── output/                          # 实验输出 (每次实验一个子目录)
│   └── <mode>_<model>_<engine>_<params>_<YYYYMMDD_HHmmss>/
│       ├── config.txt               # 实验配置记录
│       ├── *.csv                    # 结果 CSV 文件
│       └── figures/                 # 该实验的可视化图表
├── dataset/                         # 数据集
├── docs/                            # 文档
└── venv/                            # Python 虚拟环境
```

**所有命令均从项目根目录 (`dssd/`) 运行。**

### 实验输出目录

每次运行实验会在 `output/` 下自动创建独立的子目录，命名格式为：

```
<mode>_<model>_<engine>_<params>_<YYYYMMDD_HHmmss>
```

示例：
```
output/
├── token_energy_batch_Qwen3-32B_vllm_n20_t15000_r5_20260311_143052/
│   ├── config.txt                          # 完整实验参数
│   ├── token_energy_batch_per_position.csv
│   ├── token_energy_batch_step_raw.csv
│   ├── token_energy_batch_per_sample.csv
│   ├── token_energy_batch_rounds.csv
│   ├── token_energy_batch_prefill.csv
│   └── figures/
│       ├── token_energy_batch_curve.png
│       ├── token_energy_batch_comparison.png
│       ├── token_energy_cumulative.png
│       └── token_energy_distribution.png
├── token_energy_stream_Qwen3-8B_auto_n20_t12000_rate20_dur12000_w50_20260311_105903/
│   ├── config.txt
│   ├── token_energy_stream_per_position.csv
│   ├── token_energy_stream_per_sample.csv
│   ├── token_energy_stream_rounds.csv
│   └── figures/
│       ├── token_energy_batch_curve.png
│       ├── token_energy_batch_comparison.png
│       ├── token_energy_cumulative.png
│       └── token_energy_distribution.png
├── token_energy_stream_Qwen3-8B_auto_n20_t4096_rate20_dur600_mns8-16-32-64_20260312_100000/
│   ├── config.txt                          # 顶层实验配置
│   ├── batch_size_sweep_summary.csv        # 所有 batch size 的汇总对比
│   ├── mns_8/                              # max_num_seqs=8 的独立结果
│   │   ├── token_energy_stream_per_position.csv
│   │   ├── token_energy_stream_per_sample.csv
│   │   ├── token_energy_stream_rounds.csv
│   │   └── figures/
│   ├── mns_16/                             # max_num_seqs=16 的独立结果
│   │   └── ...
│   ├── mns_32/                             # max_num_seqs=32 的独立结果
│   │   └── ...
│   └── mns_64/                             # max_num_seqs=64 的独立结果
│       └── ...
└── kv_benchmark_Qwen3-32B_vllm_kv32-64-128_g64_20260311_160000/
    ├── config.txt
    ├── results_kvcache_raw.csv
    └── results_kvcache_summary.csv
```

**目录名参数含义**：
- `n20`: `--token_samples 20` (batch/sequential 的 prompt 数量; stream 的 prompt pool 大小)
- `t15000`: `--token_max_tokens 15000`
- `r5`: `--batch_repeats 5`
- `rate20`: `--req_rate 20` (stream 模式: 20 req/min)
- `dur12000`: `--duration 12000` (stream 模式: 实验时长 12000s)
- `w50`: `--warmup 50` (stream 模式: 50 个预热请求)
- `mns8-16-32-64`: `--stream_batch_sizes "8,16,32,64"` (stream 模式: 多轮 max_num_seqs sweep)
- `kv32-64-128`: `--kv_lengths 32,64,128`
- `g64`: `--gen_tokens 64`

实验结束后会自动生成可视化图表到 `figures/` 子目录中。

---

## UAV 客户端

### 基本用法

```bash
python scripts/uav_client.py [选项]
```

### 全部参数一览

#### 模型与设备

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--draft_model_name` | str | `Qwen3-1.7B` | 小模型路径 (本地路径或 HuggingFace ID) |
| `--device` | str | `auto` | 设备选择，详见下方 |
| `--gpu_ids` | str | `None` | 指定 GPU 编号，如 `0,1,2,3`，仅 `--device auto` 时有效 |
| `--framework` | str | `auto` | 推理框架: `auto` / `mlx` / `pytorch` |
| `--engine` | str | `auto` | 推理引擎: `auto` / `vllm` / `pytorch` |

**`--device` 可选值：**

| 值 | 行为 |
|----|------|
| `auto` | 自动检测: 多 GPU → 自动切分, 单 GPU → `cuda:0`, Apple → `mps`, 其他 → `cpu` |
| `cuda:0` | 指定单张 GPU |
| `cuda:0,1,2,3` | 指定多张 GPU (等效于 `--device auto --gpu_ids 0,1,2,3`) |
| `mps` | Apple Silicon Metal |
| `cpu` | CPU 推理 |

**`--engine` 与 `--framework` 的区别：**

| 参数 | 作用 | 选项 |
|------|------|------|
| `--framework` | 底层框架 | `auto` (Apple→MLX, CUDA→PyTorch), `mlx`, `pytorch` |
| `--engine` | 推理引擎 | `auto` (有 vLLM 且 NVIDIA GPU → vLLM), `vllm`, `pytorch` |

> `--engine vllm` 会自动启用 tensor parallelism，适合大模型多卡推理。
> `--engine pytorch` 使用 HuggingFace Transformers + `device_map="auto"` (pipeline parallelism)。

#### 生成参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | str | `"Alan Turing..."` | 默认输入 prompt (单次运行时使用) |
| `--max_len` | int | `256` | 每次请求最大生成 token 数 |
| `--gamma` | int | `4` | 投机解码每轮 draft 的候选 token 数 |
| `--seed` | int | `321` | 随机种子 |
| `--temperature` | float | `0.7` | 采样温度 (0 = greedy) |
| `--top_k` | int | `10` | Top-K 采样 |
| `--top_p` | float | `0` | Top-P (nucleus) 采样 (0 = 不使用) |

#### 运行模式

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--mode` | str | `all` | 运行模式，详见 [实验模式详解](#实验模式详解) |
| `--interactive` | flag | `False` | 启用交互式方法选择菜单 |

**`--mode` 可选值：**

| 模式 | 说明 | 是否需要 BS 服务器 |
|------|------|--------------------|
| `all` | 顺序运行 DSSD + DSD + Baseline + Local Baseline | ✅ 需要 |
| `dssd` | 仅运行 DSSD | ✅ 需要 |
| `dsd` | 仅运行 DSD | ✅ 需要 |
| `baseline` | 仅运行远程大模型自回归 | ✅ 需要 |
| `local_baseline` | 仅运行本地小模型自回归 | ❌ 不需要 |
| `benchmark` | 多 prompt × 多轮基准测试 | 取决于 `bench_modes` |
| `kv_benchmark` | KV Cache 长度递进测试 | ❌ 不需要 |
| `token_energy` | 逐 token 能耗精确记录 (串行) | ❌ 不需要 |
| `token_energy_batch` | 逐 token 能耗精确记录 (并发 batch) | ❌ 不需要 |
| `token_energy_stream` | 逐 token 能耗记录 (基于速率的流式请求) | ❌ 不需要 |

#### 网络与通信

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--bs_addr` | str | `127.0.0.1` | BS 服务器 IP 地址 |
| `--bs_port` | int | `50051` | BS 服务器端口 |
| `--net_type` | str | `wifi` | 网络类型 (影响网络能耗估算): `wifi` / `lte` / `eth` |

#### 输出

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--csv_path` | str | `output/results_real_network.csv` | 结果 CSV 保存路径 |

#### Benchmark 专用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num_trials` | int | `3` | 每个 prompt 重复运行次数 |
| `--num_prompts` | int | `0` | 使用内置 prompt 数量 (0 = 全部) |
| `--bench_modes` | str | `None` | 逗号分隔的方法列表，如 `dssd,dsd,local_baseline` |

#### KV Benchmark 专用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--kv_lengths` | str | `32,64,128,256,512,1024` | 逗号分隔的 KV cache 长度列表 |
| `--gen_tokens` | int | `64` | 每次固定生成的 token 数 |

#### Token Energy 专用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--token_samples` | int | `20` | prompt 数量 (sequential/batch) 或 prompt pool 大小 (stream) |
| `--token_max_tokens` | int | `128` | 每个请求最大生成 token 数 |
| `--batch_repeats` | int | `1` | 重复轮次数 (batch/stream 模式，每轮不同 seed，结果取平均) |
| `--req_rate` | float | `10.0` | ⚡ stream 专用: 请求注入速率 (requests/min)，如 20 = 每 3 秒注入一个请求 |
| `--duration` | int | `600` | ⚡ stream 专用: 每轮实验时长 (秒)，到时间后**立即结束**实验 |
| `--warmup` | int | `0` | ⚡ stream 专用: 预热请求数。计时前先注入并完成 prefill，确保 GPU 在稳定并发状态 |
| `--stream_batch_sizes` | str | `None` | ⚡ stream 专用: 逗号分隔的 vLLM `max_num_seqs` 值列表，如 `"8,16,32,64,128"`。每个值会**重建 vLLM 引擎**并跑一轮独立实验，结果分别保存 |

#### 网络限速参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--tc_enable` | flag | `False` | 启用 OS 级网络限速 (需要 sudo) |
| `--tc_profile` | str | `None` | 预设网络场景 (覆盖下面四项) |
| `--tc_bw` | str | `10mbit` | 带宽限制 |
| `--tc_delay` | str | `50ms` | 单向延迟 (RTT ≈ 2×) |
| `--tc_jitter` | str | `10ms` | 延迟抖动范围 |
| `--tc_loss` | str | `0%` | 随机丢包率 |

**预设网络场景 (`--tc_profile`)：**

| Profile | 带宽 | 延迟 | 抖动 | 丢包 | 场景 |
|---------|------|------|------|------|------|
| `wifi_good` | 50mbit | 5ms | 2ms | 0% | 良好 WiFi |
| `wifi_fair` | 10mbit | 20ms | 10ms | 1% | 一般 WiFi |
| `wifi_poor` | 2mbit | 50ms | 30ms | 5% | 差 WiFi |
| `lte_good` | 30mbit | 30ms | 5ms | 0.5% | 良好 4G |
| `lte_fair` | 5mbit | 60ms | 20ms | 2% | 一般 4G |
| `lte_poor` | 1mbit | 100ms | 40ms | 5% | 差 4G |
| `5g_mmwave` | 100mbit | 5ms | 1ms | 0% | 5G 毫米波 |
| `5g_sub6` | 50mbit | 15ms | 5ms | 0.5% | 5G Sub-6 |
| `satellite` | 5mbit | 300ms | 50ms | 3% | 卫星通信 |
| `uav_los` | 20mbit | 10ms | 5ms | 1% | UAV 视距 |
| `uav_nlos` | 5mbit | 50ms | 20ms | 3% | UAV 非视距 |
| `paper_sim` | 10mbit | 50ms | 10ms | 0% | 论文模拟设置 |

---

## 实验模式详解

### 模式 1: 单次运行 (all)

**用途**：快速验证所有方法是否正常工作。

```bash
# 需要先启动 BS 服务器
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-0.6B \
    --device auto \
    --bs_addr 127.0.0.1 \
    --mode all
```

**行为**：依次运行 DSSD → DSD → Baseline → Local Baseline，使用默认 prompt。

**输出**：每个方法的生成文本、耗时、能耗报告。

---

### 模式 2: 单方法运行

**用途**：只运行某一个方法，调试或快速测试。

```bash
# 仅本地自回归 (不需要 BS 服务器)
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-0.6B \
    --device auto \
    --mode local_baseline

# 仅 DSSD (需要 BS 服务器)
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-0.6B \
    --device cuda:0 \
    --bs_addr 192.168.1.100 \
    --mode dssd

# 仅远程大模型自回归 (需要 BS 服务器)
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-0.6B \
    --bs_addr 192.168.1.100 \
    --mode baseline
```

---

### 模式 3: 多轮基准测试 (benchmark)

**用途**：使用多个 prompt、多次重复，获得有统计意义的性能和能耗数据。

```bash
# 全方法对比 (需要 BS 服务器)
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-0.6B \
    --device auto \
    --bs_addr 127.0.0.1 \
    --mode benchmark \
    --num_trials 5 \
    --num_prompts 10

# 仅本地方法 (不需要 BS)
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-0.6B \
    --device auto \
    --mode benchmark \
    --bench_modes "local_baseline" \
    --num_trials 3

# 指定方法子集
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-0.6B \
    --device auto \
    --bs_addr 127.0.0.1 \
    --mode benchmark \
    --bench_modes "dssd,local_baseline" \
    --num_trials 3
```

**输出文件**：
- `output/results_real_network_raw.csv` — 每条 prompt × 每次 trial 的原始结果
- `output/results_real_network_summary.csv` — 按方法汇总的统计 (mean ± std)

---

### 模式 4: KV Cache 递进测试 (kv_benchmark)

**用途**：测量不同 KV cache 长度下的能耗变化，分析 compute/memory/idle 能耗随上下文长度的增长趋势。

```bash
# 基本用法
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-0.6B \
    --device auto \
    --mode kv_benchmark

# 自定义 KV cache 长度和生成 token 数
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-0.6B \
    --device auto \
    --mode kv_benchmark \
    --kv_lengths "32,64,128,256,512,1024,2048" \
    --gen_tokens 100 \
    --num_trials 5

# 大模型 + 多卡
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-32B \
    --device auto \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --mode kv_benchmark \
    --kv_lengths "32,128,512,1024" \
    --gen_tokens 50
```

**输出文件**：
- `output/results_real_network_kvcache_raw.csv` — 每个 KV 长度 × 每次 trial 的原始数据
- `output/results_real_network_kvcache_summary.csv` — 按 KV 长度汇总的统计

**关键指标**：
- `compute_energy_mj` — ALU 计算能耗 (不随 KV 长度变化)
- `memory_energy_mj` — HBM 搬运能耗 (随 KV 长度线性增长)
- `idle_energy_mj` — 静态功耗 (与运行时间成正比)

---

### 模式 5: 逐 Token 能耗记录 (token_energy)

**用途**：精确测量每个 token 位置的 GPU 能耗，分析能耗随生成位置的变化规律。

```bash
# 基本用法 (20 个 prompt, 每个 128 token)
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-0.6B \
    --device auto \
    --mode token_energy

# 增加样本量和生成长度
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-0.6B \
    --device auto \
    --mode token_energy \
    --token_samples 50 \
    --token_max_tokens 256 \
    --seed 42

# 大模型 + 多卡 + PyTorch (精确逐 token 测量)
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-32B \
    --device auto \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --engine pytorch \
    --mode token_energy \
    --token_samples 20 \
    --token_max_tokens 128

# 使用 vLLM 引擎 (需要 vLLM 0.8.1, 同步 step 支持精确测量)
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-0.6B \
    --device auto \
    --engine vllm \
    --mode token_energy \
    --token_samples 20 \
    --token_max_tokens 128
```

**数据集**：自动从三个数据集中随机采样 prompt：
- `LongForm` — 长文本生成
- `python_code_instructions_18k_alpaca` — Python 代码生成
- `WizardLM_evol_instruct_V2_196k` — 通用指令

**输出文件**（均在 `output/` 目录下）：

| 文件 | 内容 |
|------|------|
| `token_energy_per_position.csv` | 每个 token 位置的平均能耗 (mean/std/min/max/count) |
| `token_energy_per_sample.csv` | 每个 prompt 的汇总 (来源/长度/吞吐量等) |
| `token_energy_raw.csv` | 原始逐 token 数据 (sequence_idx × position → energy_mj) |

**引擎选择建议**：

| 引擎 | 精度 | 速度 | 适用场景 |
|------|------|------|----------|
| `--engine pytorch` | ✅ 精确逐 token | 🟡 较慢 (pipeline parallelism) | 需要精确的逐 token 能耗曲线 |
| `--engine vllm` (0.8.1) | ✅ 精确逐 token | ✅ 快 (tensor parallelism) | 大模型 + 多卡 + 精确测量 |

> **重要**：vLLM 0.8.1 的 `step()` 是同步的，每次 `step()` = 一次 forward pass，
> 因此可以在每个 `step()` 前后使用 Zeus `begin_window`/`end_window` 精确测量单 token 能耗。

**输出 CSV 格式示例**：

`token_energy_per_position.csv`:
```csv
position,mean_energy_mj,std_energy_mj,min_energy_mj,max_energy_mj,count
0,45.23,5.12,38.00,52.00,20
1,42.15,4.89,35.00,49.00,20
```

`token_energy_raw.csv`:
```csv
sequence_idx,position,energy_mj
0,0,45.23
0,1,42.15
1,0,48.01
```

`token_energy_per_sample.csv`:
```csv
sample_idx,source,prompt_len,generated_tokens,wall_time,throughput
0,LongForm,42,128,8.52,15.02
1,python_code,65,128,8.31,15.40
```

**预期结果解读**：

- **首 token** 能耗通常最高（GPU 从低功耗态切换 + CUDA kernel 启动开销）
- **前几个 token** 能耗逐渐稳定
- **中后段** 基本恒定或略有上升（KV cache 增长 → attention 计算量增加）
- **不同数据集**：LongForm（长 prompt，前期能耗较高）、python_code（重复模式多，能耗稳定）、WizardLM（介于两者之间）

---

### 模式 6: 并发批量逐 Token 能耗记录 (token_energy_batch)

**用途**：将所有请求同时提交给 vLLM，在同一个 batch 中执行。由于所有请求同时开始，每个 step 中所有请求的 decode position 完全对齐，因此可以精确测量每个 position 的 step 级能耗，再除以 active 请求数得到平均单 token 能耗。

**与 `token_energy` 的区别**：

| 特性 | `token_energy` (串行) | `token_energy_batch` (并发) |
|------|----------------------|----------------------------|
| 请求处理方式 | 逐个 prompt 串行生成 | 所有 prompt 在同一 batch 并发 |
| 速度 | 慢 (单请求吞吐) | 快 (batch 吞吐) |
| 能耗精度 | 精确到每个 token | 精确到每个 position (batch 平均) |
| Position 对齐 | 每条 sequence 独立 | 所有请求 position 对齐 |
| 适用场景 | 小模型 / 精确分析 | 大模型 / 快速获取 position 级能耗趋势 |

```bash
# 基本用法 (20 个请求并发, 每个 512 token, 单轮)
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-32B \
    --device auto \
    --mode token_energy_batch \
    --token_samples 20 \
    --token_max_tokens 512

# 多轮重复取平均 (5 轮, 每轮使用不同 prompt)
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-32B \
    --device auto \
    --mode token_energy_batch \
    --token_samples 20 \
    --token_max_tokens 12000 \
    --batch_repeats 5 \
    --seed 321

# 大规模测试 (8192 token, 3 轮)
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-32B \
    --device auto \
    --mode token_energy_batch \
    --token_samples 30 \
    --token_max_tokens 8192 \
    --batch_repeats 3 \
    --seed 42
```

**`--batch_repeats` 参数说明**：

- 默认值: `1` (只跑 1 轮)
- 每轮使用不同的 seed (`seed`, `seed+1`, `seed+2`, ...) 从数据集中采样不同的 prompt
- 每轮都是完整的一个 batch: 等上一轮所有请求全部完成后, 再开始下一轮
- 最终结果是所有轮次的 per-position 能耗取平均
- 多轮可以减少单次实验的随机波动, 得到更稳定的能耗曲线

**执行流程**：

1. **Phase 1 — Prefill**: vLLM 调度器逐个请求做 prefill（每个 step 处理一个请求的完整 prompt）。共 N 个 prefill steps。
2. **Phase 2 — Decode**: 所有请求进入 decode 阶段，每个 step 所有 active 请求各生成 1 个 token。Position 完全对齐。

**关键约束**：
- 所有请求必须能放入一个 batch（`num_samples ≤ max_num_seqs`，默认 256）
- KV cache 容量需足够（`num_samples × max_tokens` 不超过 KV cache 总容量）
- 不能中途插入新请求（否则 position 对齐被打乱）

**输出文件**（均在实验目录下，如 `output/token_energy_batch_Qwen3-32B_vllm_n20_t15000_r5_20260311_143052/`）：

| 文件 | 内容 |
|------|------|
| `config.txt` | 完整实验参数记录 |
| `token_energy_batch_per_position.csv` | 每个 position 的平均单 token 能耗 (多轮平均; position -1 为 prefill, position 0 标记为 prefill_tail, position 1+ 为纯 decode) |
| `token_energy_batch_step_raw.csv` | decode 阶段每个 step 的原始数据 (含 round 列, step 总能耗 + active 请求数 + per-token 能耗) |
| `token_energy_batch_prefill.csv` | 每轮 prefill 的能耗 + 最后一行为多轮平均 |
| `token_energy_batch_per_sample.csv` | 每轮每个请求的汇总 (含 round 列, 来源/prompt 长度/生成 token 数) |
| `token_energy_batch_rounds.csv` | 每轮的汇总统计 (seed/生成量/耗时/吞吐量/平均能耗) |
| `figures/` | 自动生成的可视化图表 |

**输出 CSV 格式示例**：

`token_energy_batch_per_position.csv`:
```csv
position,mean_energy_mj,std_energy_mj,min_energy_mj,max_energy_mj,count,active_requests,step_energy_mj,phase
0,15000.00,0.0,15000.00,15000.00,1,30,450000.00,prefill
1,2100.50,0.0,2100.50,2100.50,1,30,63015.00,decode
2,2105.30,0.0,2105.30,2105.30,1,30,63159.00,decode
```

`token_energy_batch_step_raw.csv`:
```csv
decode_step,position,step_energy_mj,active_requests,per_token_energy_mj
0,1,63015.00,30,2100.50
1,2,63159.00,30,2105.30
```

**预期结果解读**：

- **Position -1 (prefill)**: 能耗最高，包含 prompt 的完整 attention 计算
- **Position 0 (prefill_tail)**: prefill 完成后的第一个 decode step，能耗异常高（含 KV cache 写入、调度器切换等尾部开销），**不计入 decode 均值**
- **Position 1+** (decode): 每个 position 的能耗 = step 总能耗 / active 请求数，这是纯 decode 能耗
- **随 position 增长**: 能耗应缓慢上升（KV cache 增长 → attention 的 memory 读取量增加）
- **当请求提前结束 (EOS)**: active 请求数减少，step 总能耗下降，但 per-token 能耗不受影响

---

### 模式 7: 流式逐 Token 能耗记录 (token_energy_stream)

**用途**：模拟真实服务场景，请求**陆续到达**（而非同时开始）。每个 step 中不同请求可能处于不同的 decode position。**只记录 decode 阶段的能耗**（prefill 阶段跳过），确保结果反映纯 decode 的能耗特征。支持 **warmup 预热**：在计时开始前先注入一批请求完成 prefill，确保实验开始时 GPU 已处于稳定的高并发状态。

**与 batch 模式的对比**：

| 特性 | `token_energy_batch` (并发) | `token_energy_stream` (流式) |
|------|----------------------------|------------------------------|
| 请求到达方式 | 所有请求同时提交 | 按速率 (req/min) 陆续注入 |
| Position 对齐 | ✅ 完全对齐 | ❌ 不对齐，各请求进度不同 |
| 每个 position 的样本数 | 1 per round (所有请求共享) | 多个 (来自不同请求) |
| 实验时长控制 | 由 max_tokens 决定 | 由 `--duration` 决定 |
| Warmup 预热 | ❌ 不支持 | ✅ `--warmup N` 预注入 N 个请求 |
| Prefill 能耗 | Position 0 标记为 `prefill_tail`，不计入 decode 均值 | ❌ **跳过**，只记录 decode |
| 场景 | 理想化的 batch 推理 | 更接近真实在线服务 |
| 能耗归属 | step_energy / active_count → position | step_energy / decode_active_count → 各请求各自的 decode position |

**启动命令**：

```bash
# 基本用法: 10 req/min, 跑 10 分钟
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-32B \
    --device auto \
    --mode token_energy_stream \
    --req_rate 10 \
    --duration 600 \
    --token_max_tokens 512

# 带 warmup: 先预注入 10 个请求完成 prefill, 再开始计时实验
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-8B \
    --device auto \
    --mode token_energy_stream \
    --req_rate 15 \
    --duration 1200 \
    --token_max_tokens 12000 \
    --warmup 10

# 大规模长时实验: 20 req/min, 跑 200 分钟 (3.3 小时), 预热 50 个请求
# 推荐: 高 warmup + 长 duration 确保并发度稳定, 能清晰观察到 KV cache 效应
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-8B \
    --device auto \
    --mode token_energy_stream \
    --req_rate 20 \
    --duration 12000 \
    --token_max_tokens 12000 \
    --token_samples 20 \
    --warmup 50 \
    --seed 321

# 高负载多轮: 30 req/min, 跑 20 分钟, 3 轮重复, 预热 30 个请求
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-32B \
    --device auto \
    --mode token_energy_stream \
    --req_rate 30 \
    --duration 1200 \
    --token_max_tokens 8192 \
    --token_samples 50 \
    --batch_repeats 3 \
    --warmup 30 \
    --seed 42

# 低负载: 2 req/min, 跑 5 分钟, 观察单请求能耗 (无预热)
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-8B \
    --device auto \
    --mode token_energy_stream \
    --req_rate 2 \
    --duration 300 \
    --token_max_tokens 4096

# Batch-size sweep: 对比不同 max_num_seqs 下的能耗规律
# 每个 batch size 会重建 vLLM 引擎, 跑一轮独立的 stream 实验
# 结果分别保存到各自的子目录 (mns_8/, mns_16/, mns_32/, ...)
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-8B \
    --device auto \
    --mode token_energy_stream \
    --req_rate 20 \
    --duration 600 \
    --token_max_tokens 4096 \
    --token_samples 20 \
    --warmup 10 \
    --stream_batch_sizes "8,16,32,64,128"
```

**关键参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--req_rate` | float | `10.0` | 请求注入速率 (requests/min)，如 10 = 每 6 秒注入一个请求 |
| `--duration` | int | `600` | 每轮实验时长 (秒)，到时间后**立即结束**实验 |
| `--warmup` | int | `0` | 预热请求数。在计时开始前先注入这些请求并完成 prefill，确保实验开始时 GPU 已在稳定并发状态。设为 0 表示不预热 |
| `--stream_batch_sizes` | str | `None` | 逗号分隔的 vLLM `max_num_seqs` 值列表。每个值会**重建 vLLM 引擎**并跑一轮独立 stream 实验，结果分别保存到子目录。如 `"8,16,32,64,128"`。不设置则使用默认引擎跑单轮 |
| `--token_samples` | int | `20` | prompt pool 大小 (从 dataset 中加载)，请求循环使用 |
| `--token_max_tokens` | int | `128` | 每个请求最大生成 token 数 |
| `--batch_repeats` | int | `1` | 重复轮次数 (每轮使用不同 seed) |

> **注意**：
> - 实际注入的请求数 ≈ `warmup + req_rate × duration / 60`
> - `--warmup` 的请求在计时前完成 prefill，它们的 decode 能耗会被正常记录
> - **Prefill 能耗不计入**：每个请求的第一个 token (position 0) 是 prefill 输出，其能耗被跳过
> - `--token_samples` 控制的是 prompt pool 的大小，请求会循环使用 pool 中的 prompt
> - 使用 `--stream_batch_sizes` 时，每个 `max_num_seqs` 值会**重新创建 vLLM 引擎**（因为 `max_num_seqs` 是引擎初始化参数），每轮实验完全独立，结果保存到各自的子目录中

**输出文件**（均在实验目录下，如 `output/token_energy_stream_Qwen3-32B_vllm_n50_t512_rate10_dur600_w20_20260311_150000/`）：

| 文件 | 内容 |
|------|------|
| `config.txt` | 完整实验参数记录 |
| `token_energy_stream_per_position.csv` | 每个 position 的平均单 token **decode** 能耗 (多轮聚合, 含 mean/std/min/max/count) |
| `token_energy_stream_per_sample.csv` | 每轮每个请求的汇总 (来源/prompt 长度/生成 token 数/注入时刻/是否 warmup) |
| `token_energy_stream_steps.csv` | 每个 step 的原始记录 (step 能耗 / decode_active / prefill 数) |
| `token_energy_stream_rounds.csv` | 每轮的汇总统计 (seed/注入数/生成量/耗时/实际速率/平均能耗) |
| `figures/` | 自动生成的可视化图表 |

**使用 `--stream_batch_sizes` 时的额外输出**：

| 文件 | 内容 |
|------|------|
| `mns_8/`, `mns_16/`, ... | 每个 `max_num_seqs` 值的独立实验子目录，各含完整的 CSV 和 `figures/` |
| `batch_size_sweep_summary.csv` | 所有 batch size 的汇总对比 (decode_mean, throughput, injected, generated 等) |

**可视化图表**（自动生成到 `figures/` 子目录）：

| 文件 | 内容 |
|------|------|
| `token_energy_batch_curve.png` | 双轴图: per-token decode 能耗曲线 + 各 position 的样本数 (count)。展示能耗随 position 的变化趋势 |
| `token_energy_batch_comparison.png` | per-token 能耗 vs position 的样本数 (count)，分析能耗与并发度的关系 |
| `token_energy_cumulative.png` | 累积能耗曲线：从 position 1 到 max_position 的总能耗累积 |
| `token_energy_distribution.png` | per-token 能耗的分布直方图 (mean, median, std, CV) |

**预期结果解读**：

- **Position 0 不会出现在结果中**（prefill 被跳过），结果从 position 1 开始
- 使用 `--warmup` 后，实验开始时已有稳定的并发度，避免了前期并发爬坡导致的能耗下降假象

**`count` 列的含义**：

> ⚠️ `count` **不是某一时刻的并发度**，而是**有多少个请求曾经经过这个 position 并贡献了一个能耗样本**。
>
> 例如 `count=917` at position 5000 表示：在整个实验过程中，有 917 个不同的请求**先后**经过了 position 5000，每个请求经过时留下了一个能耗样本。但这 917 次**发生在不同时刻**，每次的并发状态都不同。
>
> - **早期 position (1~100)**：count 最多（几乎所有请求都会经过）
> - **晚期 position (>10000)**：count 较少（只有生成长文本的请求才会到达）
> - count 下降**不代表某一时刻的并发度下降**，而是**有更少的请求能走到那么远的 position**

**能耗归属机制**：

每个 `engine.step()` 的过程：
1. 测量这一个 step 消耗的总 GPU 能量 `step_energy`
2. 找出这个 step 中**实际产出了新 decode token 的请求数** `num_decode_active`
3. 均分：`per_req_energy = step_energy / num_decode_active`
4. 每个请求把分到的能耗记录到**自己当前的 decode position** 上

> 由于不同请求在同一 step 处于不同 position，能耗均分后归属到各自的 position。最终 CSV 中每个 position 的 `mean_energy_mj` 是该 position 所有样本的平均值。

**影响能耗的两个因素**：

| 因素 | 说明 | 如何区分 |
|------|------|----------|
| **KV cache 增长** | 请求在 pos 10000 时 KV cache 有 10000 个 token，attention 搬运量更大 → 单 token 能耗更高 | 这是我们要观察的目标效应 |
| **step 内并发度变化** | 同一 step 里 decode 请求数不同，`step_energy / N` 的 N 变化会影响 per-token 能耗 | 使用 `--warmup` + 高 `--req_rate` + 长 `--duration` 来稳定并发度 |

**推荐配置（稳定并发度，观察 KV cache 效应）**：

```bash
# 高 warmup + 高注入速率 + 长时间 → 并发度稳定 → KV cache 效应清晰可见
--warmup 50 --req_rate 20 --duration 12000 --token_max_tokens 12000
```

实测效果（Qwen3-8B, 8×V100）：
- 并发度 (count): 511~917，稳定度 56%
- 能耗趋势: pos 1~50 → 884 mJ, pos 5000~6000 → 1299 mJ (+52.9%), pos 10000~11000 → 1810 mJ (+113.0%)
- 线性拟合: `energy = 0.0883 × position + 806.5 mJ`，每 1000 positions 增加 88.3 mJ/token
- **结论**: 并发度稳定后，decode 能耗随 position 线性增长，反映 KV cache 搬运开销

---

## 交互式方法选择

```bash
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-0.6B \
    --device auto \
    --mode benchmark \
    --interactive
```

会显示菜单：

```
╔══════════════════════════════════════════════════════════╗
║              选择要运行的对比方法                          ║
╠══════════════════════════════════════════════════════════╣
║  1. DSSD (Distributed Split Speculative Decoding)       ║
║  2. DSD  (Distributed Speculative Decoding)             ║
║  3. Baseline — Remote LLM Autoregressive                ║
║  4. Baseline — Local SLM Autoregressive                 ║
╠══════════════════════════════════════════════════════════╣
║  输入编号 (逗号分隔), 如: 1,4                             ║
║  直接回车 = 全部运行                                      ║
╚══════════════════════════════════════════════════════════╝
```

---

## 网络限速

模拟不同网络条件下的 UAV-BS 通信：

```bash
# 使用预设场景
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-0.6B \
    --bs_addr 127.0.0.1 \
    --mode benchmark \
    --bench_modes "dssd,dsd,baseline" \
    --tc_enable \
    --tc_profile uav_nlos

# 自定义网络参数
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-0.6B \
    --bs_addr 127.0.0.1 \
    --mode all \
    --tc_enable \
    --tc_bw 5mbit \
    --tc_delay 80ms \
    --tc_jitter 20ms \
    --tc_loss 2%

# 查看所有预设场景
python scripts/uav_client.py --tc_list_profiles
```

> ⚠️ 网络限速需要 `sudo` 权限，使用 Linux `tc` (traffic control) 工具。

---

## BS 服务端

### 基本用法

```bash
python scripts/bs_server.py [选项]
```

### BS 全部参数一览

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--target_model_name` | str | `~/model_hub/Qwen3-32B` | 大模型路径 |
| `--device` | str | `auto` | 设备: `auto` (多卡自动切分), `cuda:0`, `cpu` |
| `--gpu_ids` | str | `None` | 指定 GPU, 如 `0,1,2` |
| `--cpu_offload` | flag | `False` | 启用 CPU 内存卸载 (显存不够时) |
| `--dtype` | str | `fp16` | 精度: `auto` / `fp32` / `fp16` / `bf16` |
| `--port` | int | `50051` | 监听端口 |
| `--verbose` | flag | `False` | 打印详细验证日志 |
| `--auto_restart` | flag | `False` | 崩溃后自动重启 (模型保留在显存) |
| `--max_restarts` | int | `10` | 最大自动重启次数 (0 = 无限) |

### 部署配置

```bash
# 单卡
python scripts/bs_server.py \
    --target_model_name ~/model_hub/Qwen3-32B \
    --device cuda:0 \
    --port 50051

# 多卡自动切分 (8 卡)
python scripts/bs_server.py \
    --target_model_name ~/model_hub/Qwen3-32B \
    --device auto \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --dtype bf16 \
    --port 50051

# 显存不够 → CPU offload
python scripts/bs_server.py \
    --target_model_name ~/model_hub/Qwen3-32B \
    --device auto \
    --cpu_offload \
    --port 50051

# 生产部署 (自动重启 + 详细日志)
python scripts/bs_server.py \
    --target_model_name ~/model_hub/Qwen3-32B \
    --device auto \
    --auto_restart \
    --max_restarts 0 \
    --verbose \
    --port 50051
```

---

## 可视化脚本

### KV Cache 结果可视化

```bash
# 使用默认路径 (output/results_real_network_kvcache_summary.csv)
python scripts/visualize_kvcache.py

# 指定 CSV 路径
python scripts/visualize_kvcache.py /path/to/custom_kvcache_summary.csv
```

**生成图表** (保存到 `figures/`)：

| 文件 | 内容 |
|------|------|
| `energy_vs_kvcache.png` | 各能耗组件随 KV cache 长度的变化曲线 |
| `energy_breakdown_stacked.png` | 能耗占比堆叠图 |
| `performance_vs_kvcache.png` | 吞吐量 & 延迟随 KV cache 长度变化 |
| `memory_bytes_per_token.png` | 每 token HBM 搬运量分解 |
| `energy_composition_pie.png` | 能耗组成饼图 |
| `summary_table.png` | 汇总数据表格 |

### 逐 Token 能耗可视化

> **注意**: 实验结束后会自动生成可视化图表到实验目录的 `figures/` 子目录中。
> 以下命令用于手动重新生成或对旧实验重新可视化。

```bash
# 推荐: 指定实验目录 (自动推断 data_dir 和 output_dir)
python scripts/visualize_token_energy.py \
    --experiment_dir output/token_energy_batch_Qwen3-32B_vllm_n20_t15000_r5_20260311_143052

# 或手动指定数据和输出目录
python scripts/visualize_token_energy.py \
    --data_dir output/token_energy_batch_Qwen3-32B_vllm_n20_t15000_r5_20260311_143052 \
    --output_dir output/token_energy_batch_Qwen3-32B_vllm_n20_t15000_r5_20260311_143052/figures

# 强制指定模式
python scripts/visualize_token_energy.py --experiment_dir <dir> --mode batch
python scripts/visualize_token_energy.py --experiment_dir <dir> --mode stream
python scripts/visualize_token_energy.py --experiment_dir <dir> --mode sequential
```

**Sequential 模式图表**：

| 文件 | 内容 |
|------|------|
| `token_energy_mean_curve.png` | 每个位置的平均能耗曲线 (±1σ 置信区间) |
| `token_energy_heatmap.png` | 能耗热力图 (sequence × position) |
| `token_energy_cumulative.png` | 累积能耗曲线 |
| `token_energy_sample_comparison.png` | 各 sample 的吞吐量/时间对比 |
| `token_energy_distribution.png` | 能耗分布直方图 |

**Batch 模式图表**：

| 文件 | 内容 |
|------|------|
| `token_energy_batch_curve.png` | 三合一: per-token 能耗 + step 总能耗 + active 请求数 |
| `token_energy_batch_comparison.png` | step 总能耗 vs per-token 能耗 (batch 摊薄效果) |
| `token_energy_cumulative.png` | 累积能耗曲线 |
| `token_energy_distribution.png` | 能耗分布直方图 |

**Stream 模式图表**：

| 文件 | 内容 |
|------|------|
| `token_energy_batch_curve.png` | 双轴图: per-token decode 能耗曲线 + 各 position 的样本数 (count) |
| `token_energy_batch_comparison.png` | per-token 能耗 vs position 的样本数，分析能耗与并发度关系 |
| `token_energy_cumulative.png` | 累积 decode 能耗曲线 |
| `token_energy_distribution.png` | per-token decode 能耗分布直方图 (含 mean/median/std/CV) |

> 所有图表下方均自动附带文字描述，总结关键统计数据和趋势。

---

## 多 GPU 配置

### 方式 1: 自动使用所有 GPU

```bash
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-32B \
    --device auto \
    --mode local_baseline
```

> `--device auto` 会自动检测所有可用 GPU 并使用 `device_map="auto"` 切分模型。

### 方式 2: 指定 GPU 子集

```bash
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-32B \
    --device auto \
    --gpu_ids 0,1,2,3 \
    --mode local_baseline
```

### 方式 3: 单卡

```bash
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-0.6B \
    --device cuda:0 \
    --mode local_baseline
```

### 并行方式对比

| 引擎 | 并行方式 | 特点 |
|------|----------|------|
| PyTorch (`--engine pytorch`) | Pipeline Parallelism | 模型按层切分到不同 GPU，顺序执行 |
| vLLM (`--engine vllm`) | Tensor Parallelism | 每层的矩阵运算切分到多 GPU，并行执行 |

> Tensor Parallelism 通常比 Pipeline Parallelism 更快，推荐大模型使用 vLLM。

---

## 推理引擎选择

### 自动选择逻辑 (`--engine auto`)

```
有 NVIDIA GPU?
  ├── 是 → 安装了 vLLM? → 是 → 模型兼容 vLLM? → 是 → 使用 vLLM
  │                       │                      └── 否 → 使用 PyTorch
  │                       └── 否 → 使用 PyTorch
  └── 否 → Apple Silicon? → 是 → 使用 MLX
                           └── 否 → 使用 PyTorch CPU
```

### 强制指定

```bash
# 强制 PyTorch (即使有 vLLM)
python scripts/uav_client.py --engine pytorch --device auto ...

# 强制 vLLM
python scripts/uav_client.py --engine vllm --device auto ...

# 强制 MLX (Apple Silicon)
python scripts/uav_client.py --framework mlx --device auto ...
```

---

## 典型实验配置

### 实验 1: 小模型快速验证

```bash
# 不需要 BS 服务器, 单卡即可
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-0.6B \
    --device cuda:0 \
    --mode local_baseline \
    --max_len 100
```

### 实验 2: 全方法对比 (同一台机器)

```bash
# 终端 1: 启动 BS 服务器
python scripts/bs_server.py \
    --target_model_name ~/model_hub/Qwen3-32B \
    --device auto \
    --port 50051 --verbose

# 终端 2: 启动 UAV 客户端 (全方法 benchmark)
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-0.6B \
    --device cuda:0 \
    --bs_addr 127.0.0.1 \
    --mode benchmark \
    --bench_modes "dssd,dsd,baseline,local_baseline" \
    --num_trials 5 \
    --num_prompts 10
```

### 实验 3: KV Cache 能耗分析

```bash
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-0.6B \
    --device auto \
    --mode kv_benchmark \
    --kv_lengths "32,64,128,256,512,1024,2048,4096" \
    --gen_tokens 100 \
    --num_trials 3

# 可视化
python scripts/visualize_kvcache.py
```

### 实验 4: 逐 Token 能耗曲线

```bash
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-0.6B \
    --device auto \
    --engine pytorch \
    --mode token_energy \
    --token_samples 30 \
    --token_max_tokens 256 \
    --seed 42

# 可视化
python scripts/visualize_token_energy.py
```

### 实验 5: 不同网络条件下的 DSSD 对比

```bash
# 启动 BS (终端 1)
python scripts/bs_server.py \
    --target_model_name ~/model_hub/Qwen3-32B \
    --device auto --port 50051

# 好网络 (终端 2)
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-0.6B \
    --bs_addr 127.0.0.1 \
    --mode benchmark --bench_modes "dssd,dsd,baseline" \
    --tc_enable --tc_profile wifi_good \
    --csv_path output/results_wifi_good.csv

# 差网络
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-0.6B \
    --bs_addr 127.0.0.1 \
    --mode benchmark --bench_modes "dssd,dsd,baseline" \
    --tc_enable --tc_profile uav_nlos \
    --csv_path output/results_uav_nlos.csv
```

### 实验 6: 大模型 8 卡逐 Token 能耗

```bash
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-32B \
    --device auto \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --engine vllm \
    --mode token_energy \
    --token_samples 20 \
    --token_max_tokens 128

# 可视化
python scripts/visualize_token_energy.py
```

### 实验 7: 流式 Token 能耗 — 观察 KV Cache 效应

> 推荐使用此配置观察 decode 阶段能耗随 KV cache 增长的变化趋势。
> 高 warmup + 高注入速率 + 长实验时间 = 稳定并发度 → 排除并发度干扰 → KV cache 效应清晰可见。

```bash
# 大规模长时实验 (Qwen3-8B, 8×GPU, ~3.3 小时)
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-8B \
    --device auto \
    --mode token_energy_stream \
    --req_rate 20 \
    --duration 12000 \
    --token_max_tokens 12000 \
    --token_samples 20 \
    --warmup 50 \
    --seed 321
```

**参数选择理由**：
- `--warmup 50`: 实验开始时即有 ~50 个并发请求在 decode，GPU 负载稳定
- `--req_rate 20`: 每 3 秒注入一个新请求，持续补充完成的请求，维持并发度
- `--duration 12000`: 200 分钟实验时长，共注入 ~4000 个请求，确保所有 position 有足够样本
- `--token_max_tokens 12000`: 长生成，观察到 position 12000 附近的能耗变化

### 实验 8: 流式 Token 能耗 — 不同负载对比

```bash
# 低负载 (少量并发)
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-8B \
    --device auto \
    --mode token_energy_stream \
    --req_rate 5 --duration 600 --warmup 5 \
    --token_max_tokens 4096

# 中负载
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-8B \
    --device auto \
    --mode token_energy_stream \
    --req_rate 15 --duration 1200 --warmup 20 \
    --token_max_tokens 12000

# 高负载 (大量并发)
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-8B \
    --device auto \
    --mode token_energy_stream \
    --req_rate 30 --duration 1200 --warmup 50 \
    --token_max_tokens 8192 --token_samples 50
```

> 对比不同 `req_rate` 下的能耗曲线斜率，可以分析并发度对 KV cache 能耗效应的影响。

### 实验 9: 流式 Token 能耗 — Batch Size Sweep (不同 max_num_seqs 对比)

> 通过控制 vLLM 引擎的 `max_num_seqs` 参数，限制引擎同时处理的最大请求数，
> 从而精确控制 batch size。每个 batch size 会重建引擎并跑一轮独立实验，
> 对比不同并发度下的 per-token decode 能耗和吞吐量。

```bash
# 基本 sweep: 对比 8/16/32/64/128 五种 batch size
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-8B \
    --device auto \
    --mode token_energy_stream \
    --req_rate 20 \
    --duration 600 \
    --token_max_tokens 4096 \
    --token_samples 20 \
    --warmup 10 \
    --stream_batch_sizes "8,16,32,64,128"

# 细粒度 sweep: 更多 batch size 点, 更长实验时间
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-8B \
    --device auto \
    --mode token_energy_stream \
    --req_rate 20 \
    --duration 1200 \
    --token_max_tokens 8192 \
    --token_samples 30 \
    --warmup 20 \
    --stream_batch_sizes "4,8,16,24,32,48,64,96,128"

# 大模型 sweep
python scripts/uav_client.py \
    --draft_model_name ~/model_hub/Qwen3-32B \
    --device auto \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --mode token_energy_stream \
    --req_rate 15 \
    --duration 900 \
    --token_max_tokens 4096 \
    --token_samples 20 \
    --warmup 10 \
    --stream_batch_sizes "4,8,16,32"
```

**工作原理**：

1. 解析 `--stream_batch_sizes` 为 batch size 列表，如 `[8, 16, 32, 64, 128]`
2. **跳过主进程的引擎创建**（避免浪费初始化时间）
3. 对每个 batch size `mns`，启动**独立的 Python 子进程**：
   - 子进程中 CUDA 未初始化，vLLM 可以正常使用 `fork` 模式创建 worker
   - **创建**新的 vLLM 引擎，设置 `max_num_seqs=mns`
   - 创建独立的实验子目录 `mns_{mns}/`
   - 运行完整的 stream 实验 (warmup → 速率注入 → 到时间结束)
   - 生成可视化图表到子目录的 `figures/`
   - 将 summary 写入 `_summary.json`，子进程退出
4. 主进程读取每轮的 `_summary.json`，汇总对比
5. 所有轮次完成后，输出汇总对比表并保存 `batch_size_sweep_summary.csv`

> **为什么使用子进程？** vLLM 多卡引擎使用 `fork` 模式创建 worker 进程。如果在同一进程中先销毁旧引擎再创建新引擎，CUDA 上下文已被初始化，vLLM 被迫切换到 `spawn` 模式，在 V100 等老 GPU 上容易导致 NCCL 初始化死锁。使用独立子进程可以确保每轮实验都在干净的 CUDA 环境中启动。

**预期分析**：

- **小 batch size (如 8)**：per-token 能耗较高（GPU 利用率低，idle 功耗占比大）
- **中等 batch size (如 32~64)**：per-token 能耗最低（GPU 利用率高，batch 摊薄效果最佳）
- **大 batch size (如 128+)**：per-token 能耗可能回升（HBM 带宽瓶颈，KV cache 竞争加剧）
- **吞吐量**：随 batch size 增大而增加，但增速递减（受 HBM 带宽限制）

---

## 常见问题

### Q: 如何查看当前 GPU 状态？

```bash
nvidia-smi
# 或实时监控
watch -n 1 nvidia-smi
```

### Q: 模型太大放不下显存怎么办？

1. 使用多卡: `--device auto --gpu_ids 0,1,2,3`
2. 使用 CPU offload: `--device auto` (PyTorch 会自动 offload 到 CPU)
3. 使用更小的模型

### Q: 怎么知道用了哪个推理引擎？

启动时会打印框架检测信息：

```
[DraftNode] 框架检测: pytorch (CUDA)
[DraftNode] 使用 vLLM 引擎, tensor_parallel_size=8
```

### Q: CSV 结果和图表在哪？

每次实验会在 `output/` 下自动创建独立子目录，目录名包含实验模式、模型、引擎、关键参数和时间戳。CSV 和可视化图表都保存在该目录中：

```
output/token_energy_batch_Qwen3-32B_vllm_n20_t15000_r5_20260311_143052/
├── config.txt          # 完整参数记录
├── *.csv               # 结果数据
└── figures/            # 可视化图表
```

实验结束后会自动生成可视化图表。如需手动重新生成，可以指定实验目录：

```bash
python scripts/visualize_token_energy.py --experiment_dir output/token_energy_batch_Qwen3-32B_vllm_n20_t15000_r5_20260311_143052
```
