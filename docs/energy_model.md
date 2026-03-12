# GPU 能耗计算模型文档

## 概述

本项目的 `EnergyMonitor` 模块实现了一套 **混合能耗测量方案 (Hybrid Energy Measurement)**，
结合硬件能耗计数器和分析模型，将 GPU 推理能耗拆分为三个独立部分：

| 组成部分 | 含义 | 来源 |
|---------|------|------|
| **Compute Energy (ALU)** | SM/Tensor Core 进行矩阵乘法的能耗 | FLOPs × pJ/FLOP |
| **Memory Energy (HBM)** | 从显存搬运数据到计算单元的能耗 | Bytes × pJ/Byte |
| **Idle Energy (Static)** | GPU 空闲时的静态功耗（漏电流、时钟树等） | idle_power × wall_time |

最终公式：

```
Total Energy = Compute Energy + Memory Energy + Idle Energy
```

---

## 架构总览：混合方案

### 设计原则

能耗测量的核心挑战在于：
1. **总能耗需要准确** — 不能只靠理论计算，必须有硬件实测
2. **能耗拆分需要物理意义** — 不能用 GPU 利用率做比例（利用率低 ≠ 功耗低）
3. **不同 GPU 需要适配** — H100+ 有硬件内存功率传感器，V100/A100 没有

### 三层优先级

```
┌─────────────────────────────────────────────────────────────────┐
│                    EnergyMonitor 能耗测量                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  优先级 1: Zeus 硬件能耗计数器 (推荐)                             │
│  ├── 总能耗: nvmlDeviceGetTotalEnergyConsumption                │
│  │   → GPU 内部硬件计数器, 精度最高, 不受采样频率影响               │
│  │                                                              │
│  ├── H100+ GPU: 硬件直接拆分 compute/memory                     │
│  │   └── NVML_FI_DEV_POWER_AVERAGE + NVML_POWER_SCOPE_MEMORY   │
│  │       → 直接测量 HBM 内存功率                                 │
│  │                                                              │
│  └── V100/A100: 分析模型 (Analytical Model) 拆分                │
│      └── 理论 FLOPs × pJ/FLOP : Bytes × pJ/Byte 比例           │
│                                                                 │
│  优先级 2: pynvml 周期性采样 + 积分 (退化方案)                    │
│  ├── 总能耗: Σ(gpu_power_mW × Δt) 积分                         │
│  └── 拆分: 同优先级 1 的分析模型                                 │
│                                                                 │
│  优先级 3: TDP 估算 (最后退化方案)                                │
│  ├── 总能耗: wall_time × TDP × load_factor                     │
│  └── 拆分: 同分析模型                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 为什么选择 Zeus？

| 维度 | pynvml 采样 | Zeus 硬件计数器 | 改进 |
|------|------------|----------------|------|
| **总能耗精度** | ±10-15% (受采样频率影响) | ±1-2% (硬件级) | ✅ 大幅提升 |
| **测量方式** | E ≈ Σ(P_i × Δt) | E = counter_end - counter_start | ✅ 无采样误差 |
| **对推理速度影响** | 有 (采样线程竞争) | 无 (只读计数器) | ✅ 零开销 |
| **H100+ 内存功率** | ❌ 不支持 | ✅ 支持 | ✅ 硬件直测 |
| **依赖** | pynvml | zeus-ml (内部也用 pynvml) | ≈ 相同 |
| **支持 GPU** | Kepler+ | Volta+ (V100/A100/H100) | ≈ 相同 |

Zeus 使用 NVML 的 `nvmlDeviceGetTotalEnergyConsumption` API，
这是 GPU 内部的硬件能耗计数器（类似 Intel RAPL），
比周期性采样功率再积分准确得多。

### 为什么不用 Nsight Compute (NCU)？

NCU 是 NVIDIA 的内核级 profiler，能提供极其详细的硬件指标
（如 DRAM bytes transferred、SM 利用率等），但：

| 维度 | NCU | 本项目需求 |
|------|-----|-----------|
| 工作方式 | 离线 profiling | 在线实时监控 |
| 性能影响 | 10-100x 减速 | 要求零/低开销 |
| 能耗指标 | 无直接能耗值 | 需要 mJ 级能耗 |
| 使用场景 | 内核优化 | 推理能耗对比 |
| 权限要求 | root / CAP_SYS_ADMIN | 普通用户 |

**结论**：NCU 适合离线校准 pJ/FLOP 和 pJ/Byte 常数，
但不适合作为在线能耗监控工具。可作为可选的校准手段。

---

## Zeus 集成细节

### 安装

```bash
pip install zeus-ml
```

> Zeus 内部依赖 `pynvml`，安装 `zeus-ml` 会自动安装。

### 初始化流程

```python
# energy_monitor.py 中的初始化
from zeus.monitor.energy import ZeusMonitor
from zeus.device.gpu import get_gpus

# 1. 创建 ZeusMonitor (指定要监控的 GPU)
monitor = ZeusMonitor(gpu_indices=[0])

# 2. 获取 GPU 对象 (用于查询内存功率等)
gpus_mgr = get_gpus()
gpu_obj = gpus_mgr._gpus[0]

# 3. 检查是否支持硬件能耗计数器 (Volta+)
assert gpu_obj.supportsGetTotalEnergyConsumption()

# 4. 检查是否支持 HBM 内存功率直测 (H100+ only)
try:
    mem_power = gpu_obj.getAverageMemoryPowerUsage()  # mW
    supports_mem_power = mem_power > 0
except:
    supports_mem_power = False
```

### 测量流程

```python
# 开始测量
monitor.begin_window("inference_run_1")

# ... 推理代码 ...

# 结束测量
result = monitor.end_window("inference_run_1")

# result.gpu_energy: dict[int, float]  → GPU 索引 → 能耗 (Joules)
# result.time: float                   → 经过时间 (秒)
total_gpu_energy_j = result.gpu_energy[0]  # GPU 0 的能耗 (J)
```

### H100+ 内存功率

在 H100 及更新的 GPU 上，NVML 支持直接测量 HBM 内存功率：

```python
# 仅 H100+ 支持
mem_power_mw = gpu_obj.getAverageMemoryPowerUsage()  # 单位: mW
# V100/A100 会抛出 NVMLError (Not Supported)
```

这使得在 H100+ 上可以直接从硬件获得 memory 能耗，
无需依赖分析模型的比例拆分。

---

## 分析模型 (Analytical Model)

### 核心思想

Decoder-only Transformer 在自回归推理（decode 阶段，batch_size=1）时，
每生成 1 个 token 需要：

1. **计算量 (FLOPs)**：固定，与序列长度无关
2. **显存搬运量 (Bytes)**：随序列长度增长（因为 KV cache 越来越大）

分析模型用来确定 compute 和 memory 的 **理论比例**，
然后用这个比例去拆分实测的动态能耗。

### 每 token 的计算量 (FLOPs)

```
FLOPs_per_token = 2 × P
```

其中：
- `P` = 模型总参数量
- 每个参数做 1 次 MAC (Multiply-Accumulate) = 2 FLOPs（1 乘 + 1 加）

> 注：这是 decode 阶段的估算。Prefill 阶段 FLOPs 与 seq_len 成正比，
> 但本项目主要关注 decode 阶段的逐 token 生成。

### 每 token 的显存搬运量 (Bytes) — 四部分拆分

#### ① 权重搬运 (Weight Loading)

```
weight_bytes = P × b
```

- 每次 forward pass 需要从 HBM 搬运全部模型权重到 SM
- `b` = 每参数字节数（FP16 = 2, INT8 = 1, FP32 = 4）
- **注意**：这里的"权重搬运"是指 decode 阶段每生成 1 个 token 时，
  GPU 需要从 HBM 读取全部权重到 SM 做矩阵-向量乘法。
  **不是**模型加载（模型加载只在启动时发生一次，不计入能耗监控）。
  由于模型权重远超 GPU L2 cache 容量（V100 L2 = 6MB, A100 = 40MB），
  每次 forward pass 都需要重新从 HBM 读取。

#### ② KV Cache 读取 (Attention 读历史)

```
kv_read_bytes = 2 × L × d_kv × seq_len × b
```

- Attention 需要读取所有历史 token 的 Key 和 Value
- `L` = `num_hidden_layers`（Transformer 层数）
- `d_kv` = `num_key_value_heads × head_dim`（每层 KV 的总维度）
- `seq_len` = 当前上下文长度（随生成逐渐增长）
- `2` = K 和 V 两个矩阵
- **关键**：这是唯一随 `seq_len` 增长的部分

> 使用 GQA (Grouped-Query Attention) 的模型，`num_key_value_heads` 远小于
> `num_attention_heads`，KV cache 读取量大幅减少。
> 例如 Qwen3-32B 有 40 个 attention heads 但只有 8 个 KV heads。

#### ③ KV Cache 写入 (写新 token 的 K,V)

```
kv_write_bytes = 2 × L × d_kv × 1 × b
```

- 每个新 token 只写 1 组 K,V 对
- 相对较小（只写 1 个 token vs 读 seq_len 个 token）

#### ④ 激活值搬运 (Hidden States + FFN)

```
activation_bytes = 2 × L × (d_model + d_inter) × b
```

- `d_model` = `hidden_size`（隐藏层维度）
- `d_inter` = `intermediate_size`（FFN 中间层维度）
- `2` = 读和写
- 每层的 hidden state 和 FFN 中间结果需要在 HBM 和 SM 之间搬运

#### 总搬运量

```
total_bytes_per_token = ① + ② + ③ + ④
                      = P×b + 2·L·d_kv·seq_len·b + 2·L·d_kv·b + 2·L·(d_model+d_inter)·b
```

### 关键特性：比例随 seq_len 变化

以 Qwen3-0.6B (FP16) 为例：

| seq_len | 权重(MB) | KV读(MB) | 总(MB) | compute% | memory% |
|---------|----------|----------|--------|----------|---------|
| 32      | 1200.0   | 1.8      | 1202.3 | 3.3%     | 96.7%   |
| 128     | 1200.0   | 7.2      | 1207.6 | 3.3%     | 96.7%   |
| 512     | 1200.0   | 28.7     | 1229.1 | 3.2%     | 96.8%   |
| 1024    | 1200.0   | 57.3     | 1257.8 | 3.2%     | 96.8%   |
| 4096    | 1200.0   | 229.4    | 1429.8 | 2.8%     | 97.2%   |

特性：
- **短序列**时权重搬运占主导
- **长序列**时 KV cache 读取量追上来，memory 占比进一步提高
- **大模型**的权重搬运占绝对主导（因为参数量大）

---

## 能耗拆分流程

### 完整流程图

```
                    ┌──────────────┐
                    │  Zeus 可用?   │
                    └──────┬───────┘
                      yes  │  no
              ┌────────────┴────────────┐
              │                         │
    ┌─────────▼─────────┐    ┌──────────▼──────────┐
    │ Zeus 硬件计数器     │    │ pynvml 采样可用?     │
    │ 总能耗 (最准确)     │    └──────────┬──────────┘
    └─────────┬─────────┘         yes    │  no
              │               ┌──────────┴──────────┐
              │               │                     │
    ┌─────────▼─────────┐  ┌──▼──────────────┐  ┌──▼──────────────┐
    │ H100+ GPU?         │  │ pynvml 积分     │  │ TDP × wall_time │
    └─────────┬─────────┘  │ Σ(P_i × Δt)     │  │ × load_factor   │
        yes   │  no        └──────┬──────────┘  └──────┬──────────┘
    ┌─────────┴─────────┐         │                     │
    │                   │         │                     │
┌───▼──────────┐  ┌─────▼──────┐  │                     │
│ 硬件直接拆分  │  │ 分析模型    │  │                     │
│ HBM 功率传感器│  │ 比例拆分    │  │                     │
└──────────────┘  └────────────┘  │                     │
                                  │                     │
                        ┌─────────▼─────────┐           │
                        │ 分析模型比例拆分    │◄──────────┘
                        │ (compute:memory)  │
                        └───────────────────┘
```

### 空闲功率基线

GPU 即使空闲也有 **静态功耗**（漏电流、时钟树、PHY 接口等），
典型值约为 TDP 的 10-20%。

实现方式：
1. **推理前测量空闲功率**：在 `start()` 之前，采样 5 次 GPU 功率取中位数
2. **计算静态能耗**：`idle_energy = idle_power × wall_time`
3. **计算动态能耗**：`dynamic_energy = total_energy - idle_energy`
4. **只对动态能耗做拆分**：`compute = dynamic × compute_ratio`，`memory = dynamic × memory_ratio`

```
Total Energy (Zeus 实测 或 pynvml 积分)
    ├── Idle Energy = idle_power × wall_time      ← 不拆分
    └── Dynamic Energy = Total - Idle
            ├── Compute Energy = Dynamic × compute_ratio
            └── Memory Energy  = Dynamic × memory_ratio
```

### 能耗常数 (pJ/FLOP 和 pJ/Byte)

不同 GPU 架构有不同的能效特性：

| GPU 系列 | 工艺 | pJ/FLOP | pJ/Byte | 显存类型 |
|----------|------|---------|---------|---------|
| V100 (Volta) | 12nm | 0.4 | 20.0 | HBM2 |
| A100 (Ampere) | 7nm | 0.3 | 16.0 | HBM2e |
| H100 (Hopper) | 4nm | 0.15 | 12.0 | HBM3 |
| H200 | 4nm | 0.14 | 11.0 | HBM3e |
| L40 (Ada) | 4nm | 0.25 | 25.0 | GDDR6X |
| RTX 4090 (Ada) | 4nm | 0.25 | 25.0 | GDDR6X |
| Jetson Orin | - | 0.5 | 30.0 | LPDDR5 |

> 这些值来自公开的学术论文和硬件规格书。
> GDDR 的 pJ/Byte 高于 HBM，因为 GDDR 的能效较低。

推导方法：
```
pJ_per_FLOP ≈ (TDP × ALU占比) / peak_TFLOPS
  例: V100 TDP=300W, ALU占比~60%, FP16 peak=125 TFLOPS
  → 300×0.6 / 125e12 × 1e12 ≈ 1.44 pJ (标量)
  → Tensor Core 效率更高, 有效值 ~0.4 pJ

pJ_per_byte ≈ HBM功耗 / 带宽
  例: V100 HBM2 功耗~60W, 带宽 900 GB/s
  → 60 / 900e9 × 1e12 ≈ 67 pJ (含控制器开销)
  → 纯数据搬运 ~20 pJ/byte
```

### 理论比例计算

```
theory_compute = total_flops × pJ_per_FLOP / 10⁹    (pJ → mJ)
theory_memory  = total_bytes × pJ_per_Byte / 10⁹    (pJ → mJ)

compute_ratio = theory_compute / (theory_compute + theory_memory)
memory_ratio  = theory_memory  / (theory_compute + theory_memory)
```

这些比例只用来拆分动态能耗，实际总能耗来自 Zeus 或 pynvml 实测。

---

## 模型结构参数的获取

从 `model.config` 自动提取以下参数：

| 参数 | 含义 | 示例 (Qwen3-32B) |
|------|------|------------------|
| `num_hidden_layers` | Transformer 层数 | 64 |
| `hidden_size` | 隐藏层维度 | 5120 |
| `intermediate_size` | FFN 中间层维度 | 27648 |
| `num_attention_heads` | 注意力头数 | 40 |
| `num_key_value_heads` | KV 头数 (GQA) | 8 |
| `head_dim` | 每头维度 | 128 |

支持多种命名约定：
- Qwen/LLaMA/Mistral: `hidden_size`, `num_attention_heads`, ...
- GPT-2/GPT-Neo: `n_embd`, `n_head`, `n_layer`, `n_inner`

如果无法获取 `model.config`（例如非 HuggingFace 模型），
退回到粗糙估算（只算权重搬运，不算 KV cache）。

---

## 代码调用流程

```python
# 1. 创建 EnergyMonitor，传入 model
energy_mon = EnergyMonitor(
    device=uav_node.device,
    framework=uav_node.framework,
    model=uav_node.model          # ← 用于提取 config 和参数量
)

# 2. start() 会自动:
#    - 测量空闲功率基线
#    - 初始化 Zeus 测量窗口 (如果 Zeus 可用)
#    - 启动 pynvml 采样线程 (作为辅助参考)
energy_mon.start()

# 3. ... 推理代码 ...

# 4. stop() 传入 tokens_generated 和 avg_seq_len
avg_seq_len = initial_len + total_tokens // 2
energy_stats = energy_mon.stop(
    tokens_generated=total_tokens,
    avg_seq_len=avg_seq_len       # ← 用于 KV cache 读取量估算
)

# 5. 格式化输出
print(EnergyMonitor.format_report(energy_stats, total_tokens))
```

### avg_seq_len 的含义

Decode 阶段的上下文长度从 `prompt_len` 增长到 `prompt_len + tokens_generated`。
`avg_seq_len` 是这个过程的平均值，用于估算 KV cache 的平均读取量。

```
avg_seq_len ≈ prompt_len + tokens_generated / 2
```

如果调用方不传 `avg_seq_len`，`stop()` 会自动估算为 `tokens_generated / 2`。

---

## 输出示例

### V100 (Zeus 硬件计数器 + 分析模型拆分)

```
  [进程级指标 — 仅本进程, 不受其他应用影响]
  ⏱  Wall time:          3.02s
  🖥  CPU time (process): 2.10s (user 2.00s + sys 0.10s)
  📈 CPU utilization:     69.8%
  💾 Peak memory (RSS):   3672.1 MB
  🎮 Peak GPU memory:     1153.2 MB
  📊 Avg GPU SM util:     19.6% (参考)
  📊 Avg GPU Mem BW util: 3.8% (参考)

  [设备能耗 — Zeus 硬件计数器 + Analytical Model]
  🔧 Compute type:        GPU 密集型 (含 GPU 能耗)
  ⚡ Device TDP:          300000 mW
  📐 Est. method:         zeus_hw_total_analytical_refined (...)
  🎮 GPU:                 Tesla V100-SXM2-32GB
  📋 Energy specs:        pJ/FLOP=0.4, pJ/Byte=20.0
  💤 Idle power baseline: 59988 mW

  [分析模型 — 每 token 操作量]
  🧠 Model:               0.60B params, 2 bytes/param
  📐 FLOPs/token:         1.19e+09 (2 × P)
  📐 Bytes/token (HBM):   1.20e+09 (avg_seq_len=30)
      ① 权重搬运:         1.19e+09 (1192.1 MB, P×b)
      ② KV cache 读:      3.44e+06 (3.4 MB, 2·L·d_kv·seq·b)
      ③ KV cache 写:      1.15e+05 (114.7 KB, 2·L·d_kv·1·b)
      ④ 激活值搬运:       4.59e+05 (458.8 KB, 2L(d+d_ff)b)
  📐 Theory compute:      23.842 mJ (FLOPs × 0.4 pJ/FLOP)
  📐 Theory memory:       1196.114 mJ (Bytes × 20.0 pJ/Byte)
  📐 Theory ratio:        compute 2.0% : memory 98.0%

  [能耗拆分 — 总能耗 = compute + memory + idle]
  ⚙️  Compute energy (ALU): 2702 mJ (2.701 J)
  💿 Memory energy (HBM):  135530 mJ (135.530 J)
  💤 Idle energy (static): 24394 mJ (24.394 J)
  ─────────────────────────────────────
  🔋 Device total energy:  162625 mJ (162.62 J)
  🔋 Device avg power:     53912 mW
  📊 Breakdown:            compute 1.7% | memory 83.3% | idle 15.0%

  [汇总]
  🔋 Total energy:        162625 mJ (162.62 J)
     (compute 2702 + memory 135530 + idle 24394 mJ)
  🔋 Energy/token:        3252.5 mJ/token

  [硬件级参考 — 全系统, 可能含其他应用 (nvml)]
  ⚡ Sys avg power:       64758 mW
  ⚡ Sys peak power:      66802 mW
  🔋 Sys total energy:    195342 mJ (195.34 J)

  🚀 Throughput:          16.58 tokens/s
```

> 注意：Zeus 总能耗 (162.6 J) 比 pynvml 采样积分 (195.3 J) 更低且更准确。
> pynvml 采样会因为采样时间对齐和频率问题导致高估。

---

## 退化行为

| 场景 | 总能耗来源 | 拆分方式 | 空闲功率 |
|------|-----------|---------|---------|
| Zeus 可用 + H100+ | Zeus 硬件计数器 | 硬件 HBM 功率 | 实测 |
| Zeus 可用 + V100/A100 | Zeus 硬件计数器 | 分析模型比例 | 实测 |
| Zeus 不可用 + pynvml | pynvml 采样积分 | 分析模型比例 | 实测 |
| 无 NVML (macOS/CPU) | TDP × wall_time | 分析模型比例 | 无 |

---

## 参考文献

- [Zeus: Understanding and Optimizing GPU Energy Consumption of DNN Training](https://ml.energy/zeus) (University of Michigan)
- Horowitz, "1.1 Computing's Energy Problem", ISSCC 2014
- Jouppi et al., "In-Datacenter Performance Analysis of a Tensor Processing Unit" (Google TPU)
- Jouppi et al., "TPU v4: An Optically Reconfigurable Supercomputer", ISCA 2023
- NVIDIA, "GPU Architecture Whitepapers" (V100, A100, H100)
- Patterson et al., "Carbon Emissions and Large Neural Network Training" (Google, 2021)
- Desislavov et al., "Trends in AI inference energy consumption" (2023)
- Leng et al., "GPUWattch: Enabling Energy Optimizations in GPGPUs", ISCA 2013
- NVIDIA NVML API Reference

---

> **实验启动方式、命令行参数、常见问题** 请参阅 [startup_guide.md](startup_guide.md)。
