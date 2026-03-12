#!/usr/bin/env python3
"""
visualize_token_energy.py - 逐 token 能耗可视化

读取 token_energy_per_position.csv, token_energy_per_sample.csv,
token_energy_raw.csv, 生成可视化图表。

用法:
    python visualize_token_energy.py [--data_dir ./] [--output_dir ./figures]

生成图表:
  1. 每个 token 位置的平均能耗曲线 (带置信区间)
  2. 能耗分布热力图 (position × energy)
  3. 每个 sample 的总能耗 / 吞吐量对比
  4. 能耗随 token 位置的累积曲线
  5. 汇总统计表
"""

import argparse
import os
import csv
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("ERROR: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


def read_csv(path):
    """读取 CSV 文件, 返回 list of dict。"""
    if not os.path.exists(path):
        print(f"⚠ File not found: {path}")
        return []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_mean_energy_curve(pos_data, output_dir):
    """
    图1: 每个 token 位置的平均能耗曲线 (带 std 置信区间)。
    """
    if not pos_data:
        return

    positions = [int(r["position"]) for r in pos_data]
    means = [float(r["mean_energy_mj"]) for r in pos_data]
    stds = [float(r["std_energy_mj"]) for r in pos_data]
    counts = [int(r["count"]) for r in pos_data]

    means = np.array(means)
    stds = np.array(stds)
    positions = np.array(positions)

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # 主曲线: 平均能耗
    ax1.plot(positions, means, color="#2196F3", linewidth=1.5,
             label="Mean energy per token", zorder=3)
    ax1.fill_between(positions, means - stds, means + stds,
                     alpha=0.2, color="#2196F3", label="±1 std", zorder=2)

    ax1.set_xlabel("Token Position", fontsize=12)
    ax1.set_ylabel("Energy (mJ)", fontsize=12, color="#2196F3")
    ax1.tick_params(axis='y', labelcolor="#2196F3")
    ax1.set_title("Per-Token Energy Consumption by Position", fontsize=14, pad=15)
    ax1.grid(True, alpha=0.3)

    # 次轴: 样本数
    ax2 = ax1.twinx()
    ax2.bar(positions, counts, alpha=0.15, color="#FF9800",
            width=1.0, label="Sample count", zorder=1)
    ax2.set_ylabel("Sample Count", fontsize=12, color="#FF9800")
    ax2.tick_params(axis='y', labelcolor="#FF9800")

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    path = os.path.join(output_dir, "token_energy_mean_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path}")


def plot_energy_heatmap(raw_data, output_dir):
    """
    图2: 能耗分布热力图 — 每条 sequence 的逐 token 能耗。
    """
    if not raw_data:
        return

    # 解析原始数据
    seq_indices = set()
    max_pos = 0
    for r in raw_data:
        seq_indices.add(int(r["sequence_idx"]))
        max_pos = max(max_pos, int(r["position"]))

    n_seqs = len(seq_indices)
    if n_seqs == 0:
        return

    # 构建矩阵
    matrix = np.full((n_seqs, max_pos + 1), np.nan)
    for r in raw_data:
        si = int(r["sequence_idx"])
        pos = int(r["position"])
        energy = float(r["energy_mj"])
        matrix[si, pos] = energy

    fig, ax = plt.subplots(figsize=(14, max(4, n_seqs * 0.3 + 1)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")
    ax.set_xlabel("Token Position", fontsize=12)
    ax.set_ylabel("Sequence Index", fontsize=12)
    ax.set_title("Token Energy Heatmap (mJ)", fontsize=14, pad=15)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Energy (mJ)", fontsize=10)

    plt.tight_layout()
    path = os.path.join(output_dir, "token_energy_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path}")


def plot_cumulative_energy(pos_data, output_dir):
    """
    图3: 累积能耗曲线 — 生成 N 个 token 总共消耗多少能量。
    """
    if not pos_data:
        return

    positions = [int(r["position"]) for r in pos_data]
    means = [float(r["mean_energy_mj"]) for r in pos_data]

    cumulative = np.cumsum(means)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(positions, cumulative, color="#4CAF50", linewidth=2,
            label="Cumulative energy")
    ax.fill_between(positions, 0, cumulative, alpha=0.15, color="#4CAF50")

    ax.set_xlabel("Token Position", fontsize=12)
    ax.set_ylabel("Cumulative Energy (mJ)", fontsize=12)
    ax.set_title("Cumulative Energy Consumption vs. Token Position", fontsize=14, pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 标注总能耗
    total = cumulative[-1] if len(cumulative) > 0 else 0
    ax.annotate(f"Total: {total:.1f} mJ\n({total/1000:.3f} J)",
                xy=(positions[-1], cumulative[-1]),
                xytext=(-80, -30), textcoords="offset points",
                fontsize=10, color="#4CAF50",
                arrowprops=dict(arrowstyle="->", color="#4CAF50"))

    # 底部文字描述
    desc_lines = []
    n_pos = len(positions)
    total_j = total / 1000
    desc_lines.append(
        f"Cumulative energy to generate {n_pos} tokens (per-position mean): "
        f"{total:.0f} mJ ({total_j:.3f} J)")
    if n_pos > 1:
        avg_per_token = total / n_pos
        desc_lines.append(f"Average cost: {avg_per_token:.1f} mJ/token")
        # 斜率分析: 前半 vs 后半
        half = n_pos // 2
        first_half_cost = cumulative[half - 1] if half > 0 else 0
        second_half_cost = total - first_half_cost
        desc_lines.append(
            f"First half (pos 1~{positions[half-1]}): {first_half_cost:.0f} mJ | "
            f"Second half (pos {positions[half]}~{positions[-1]}): {second_half_cost:.0f} mJ")
        if first_half_cost > 0:
            ratio = second_half_cost / first_half_cost
            slope_note = "steeper" if ratio > 1.05 else \
                         "flatter" if ratio < 0.95 else "similar"
            desc_lines.append(
                f"Second/First half ratio: {ratio:.2f}x — "
                f"later tokens are {slope_note} in energy cost")
    desc = "\n".join(desc_lines)

    fig.text(0.5, -0.02, desc, ha="center", va="top", fontsize=10,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                       edgecolor="#BDBDBD", alpha=0.9))

    plt.tight_layout()
    path = os.path.join(output_dir, "token_energy_cumulative.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path}")


def plot_sample_comparison(sample_data, output_dir):
    """
    图4: 每个 sample 的吞吐量和总生成 token 数。
    """
    if not sample_data:
        return

    indices = list(range(len(sample_data)))
    sources = [r["source"] for r in sample_data]
    throughputs = [float(r["throughput"]) for r in sample_data]
    gen_tokens = [int(r["generated_tokens"]) for r in sample_data]
    wall_times = [float(r["wall_time"]) for r in sample_data]

    # 颜色映射: 不同数据集不同颜色
    source_colors = {
        "LongForm": "#2196F3",
        "python_code": "#FF9800",
        "WizardLM": "#4CAF50",
    }
    colors = [source_colors.get(s, "#9E9E9E") for s in sources]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 左图: 吞吐量
    bars1 = ax1.bar(indices, throughputs, color=colors, alpha=0.8)
    ax1.set_xlabel("Sample Index", fontsize=12)
    ax1.set_ylabel("Throughput (tokens/s)", fontsize=12)
    ax1.set_title("Throughput per Sample", fontsize=13)
    ax1.grid(True, alpha=0.3, axis='y')

    # 右图: 生成时间
    bars2 = ax2.bar(indices, wall_times, color=colors, alpha=0.8)
    ax2.set_xlabel("Sample Index", fontsize=12)
    ax2.set_ylabel("Wall Time (s)", fontsize=12)
    ax2.set_title("Generation Time per Sample", fontsize=13)
    ax2.grid(True, alpha=0.3, axis='y')

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, alpha=0.8, label=s)
                       for s, c in source_colors.items()
                       if s in sources]
    ax1.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    path = os.path.join(output_dir, "token_energy_sample_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path}")


def plot_energy_distribution(pos_data, output_dir):
    """
    图5: 能耗分布直方图 — 所有 token 的能耗分布。
    """
    if not pos_data:
        return

    means = [float(r["mean_energy_mj"]) for r in pos_data]
    mins = [float(r["min_energy_mj"]) for r in pos_data]
    maxs = [float(r["max_energy_mj"]) for r in pos_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(means, bins=30, color="#2196F3", alpha=0.7,
            edgecolor="white", label="Mean energy")
    overall_mean = np.mean(means)
    ax.axvline(overall_mean, color="#F44336", linestyle="--",
               linewidth=2, label=f"Overall mean: {overall_mean:.2f} mJ")

    ax.set_xlabel("Energy per Token (mJ)", fontsize=12)
    ax.set_ylabel("Count (positions)", fontsize=12)
    ax.set_title("Distribution of Per-Token Energy", fontsize=14, pad=15)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 底部文字描述
    desc_lines = []
    std_val = np.std(means)
    median_val = np.median(means)
    desc_lines.append(
        f"Histogram of per-position mean energy across all {len(means)} decode positions.")
    desc_lines.append(
        f"Mean: {overall_mean:.1f} mJ | Median: {median_val:.1f} mJ | "
        f"Std: {std_val:.1f} mJ | Range: [{min(means):.0f}, {max(means):.0f}] mJ")
    # 判断分布形态
    skew = (overall_mean - median_val) / std_val if std_val > 0 else 0
    cv = std_val / overall_mean * 100 if overall_mean > 0 else 0
    if cv < 5:
        shape = "Very concentrated — energy is nearly constant across positions."
    elif cv < 15:
        shape = "Moderately spread — some variation across positions."
    else:
        shape = "Widely spread — significant energy variation across positions."
    desc_lines.append(f"CV = {cv:.1f}%. {shape}")
    desc = "\n".join(desc_lines)

    fig.text(0.5, -0.02, desc, ha="center", va="top", fontsize=10,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                       edgecolor="#BDBDBD", alpha=0.9))

    plt.tight_layout()
    path = os.path.join(output_dir, "token_energy_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path}")


def print_summary_table(pos_data, sample_data):
    """打印汇总统计表。"""
    if not pos_data:
        print("  ⚠ No position data available.")
        return

    means = [float(r["mean_energy_mj"]) for r in pos_data]
    stds = [float(r["std_energy_mj"]) for r in pos_data]

    total_positions = len(pos_data)
    overall_mean = np.mean(means)
    overall_std = np.mean(stds)
    overall_min = min(float(r["min_energy_mj"]) for r in pos_data)
    overall_max = max(float(r["max_energy_mj"]) for r in pos_data)
    cumulative = sum(means)

    print(f"\n{'='*60}")
    print(f"  TOKEN-LEVEL ENERGY SUMMARY")
    print(f"{'='*60}")
    print(f"  Total positions:        {total_positions}")
    print(f"  Mean energy/token:      {overall_mean:.4f} mJ")
    print(f"  Avg std across pos:     {overall_std:.4f} mJ")
    print(f"  Min (any pos):          {overall_min:.4f} mJ")
    print(f"  Max (any pos):          {overall_max:.4f} mJ")
    print(f"  Cumulative ({total_positions} tokens): {cumulative:.2f} mJ "
          f"({cumulative/1000:.4f} J)")

    if sample_data:
        n_samples = len(sample_data)
        avg_throughput = np.mean([float(r["throughput"]) for r in sample_data])
        avg_wall_time = np.mean([float(r["wall_time"]) for r in sample_data])

        # 按数据集统计
        from collections import Counter
        source_dist = Counter(r["source"] for r in sample_data)

        print(f"\n  Samples:                {n_samples}")
        print(f"  Avg throughput:         {avg_throughput:.2f} tokens/s")
        print(f"  Avg wall time:          {avg_wall_time:.2f} s")
        print(f"  Dataset distribution:   {dict(source_dist)}")

    # 分段统计 (前10, 中间, 后10)
    if total_positions >= 20:
        first_10 = np.mean(means[:10])
        last_10 = np.mean(means[-10:])
        mid_start = total_positions // 2 - 5
        mid_10 = np.mean(means[mid_start:mid_start+10])

        print(f"\n  Position-wise breakdown:")
        print(f"    First 10 tokens:      {first_10:.4f} mJ/token")
        print(f"    Middle 10 tokens:     {mid_10:.4f} mJ/token")
        print(f"    Last 10 tokens:       {last_10:.4f} mJ/token")
        ratio = last_10 / first_10 if first_10 > 0 else 0
        print(f"    Last/First ratio:     {ratio:.2f}x")

    print(f"{'='*60}")


# ============ Batch Mode Visualizations ============

def _build_curve_description(positions, means, stds, counts_or_active,
                              step_energy, is_stream=False):
    """根据数据自动生成三合一曲线图的文字描述。"""
    lines = []

    # 基本信息
    max_pos = int(positions[-1]) if len(positions) > 0 else 0
    n_positions = len(positions)
    lines.append(f"Position range: 1 ~ {max_pos} ({n_positions} decode positions)")

    # per-token 能耗趋势
    if n_positions >= 20:
        early = np.mean(means[:min(50, n_positions)])
        late = np.mean(means[-min(50, n_positions):])
        mid_s = n_positions // 2 - 25
        mid_e = n_positions // 2 + 25
        mid = np.mean(means[max(0, mid_s):min(n_positions, mid_e)])
        trend = "increasing" if late > early * 1.03 else \
                "decreasing" if late < early * 0.97 else "stable"
        lines.append(f"Per-token energy: early={early:.0f}, mid={mid:.0f}, "
                      f"late={late:.0f} mJ  (trend: {trend})")
        lines.append(f"Overall mean: {np.mean(means):.1f} +/- {np.mean(stds):.1f} mJ/token")

    # count / active 请求数变化
    if counts_or_active is not None and len(counts_or_active) > 0:
        label = "Sample count" if is_stream else "Active requests"
        lines.append(f"{label}: {int(counts_or_active[0])} -> {int(counts_or_active[-1])} "
                      f"(peak={int(np.max(counts_or_active))})")

    # step 总能耗
    if step_energy is not None and np.any(step_energy > 0):
        lines.append(f"Step energy: mean={np.mean(step_energy):.0f} mJ, "
                      f"total={np.sum(step_energy)/1000:.1f} J")

    return "\n".join(lines)


def plot_batch_energy_curve(pos_data, output_dir):
    """
    Batch 模式: per-position 平均 token 能耗曲线, 带 active 请求数。
    """
    if not pos_data:
        return

    positions = np.array([int(r["position"]) for r in pos_data])
    means = np.array([float(r["mean_energy_mj"]) for r in pos_data])
    stds = np.array([float(r.get("std_energy_mj", 0)) for r in pos_data])

    # 兼容 batch (active_requests) 和 stream (count) 两种 CSV 格式
    has_active = "active_requests" in pos_data[0]
    has_step_energy = "step_energy_mj" in pos_data[0]

    if has_active:
        counts_or_active = np.array([int(r["active_requests"]) for r in pos_data])
        count_label = "Active Requests"
        count_legend = "Active requests"
    else:
        counts_or_active = np.array([int(r.get("count", 1)) for r in pos_data])
        count_label = "Sample Count"
        count_legend = "Sample count (requests reaching this position)"

    step_energy = np.array([float(r["step_energy_mj"]) for r in pos_data]) \
        if has_step_energy else np.zeros(len(pos_data))

    is_stream = not has_active

    # 生成描述文字
    desc = _build_curve_description(positions, means, stds, counts_or_active,
                                     step_energy if has_step_energy else None,
                                     is_stream=is_stream)

    # 决定布局: 有 step_energy 时画三图, 否则画两图
    if has_step_energy:
        fig, axes = plt.subplots(3, 1, figsize=(14, 16), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 2, 1]})
    else:
        fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})

    # 上图: per-token 平均能耗
    ax1 = axes[0]
    ax1.plot(positions, means, color="#2196F3", linewidth=1.5,
             label="Mean energy per token (mJ)")
    if stds.any():
        ax1.fill_between(positions, means - stds, means + stds,
                         alpha=0.15, color="#2196F3", label="\u00b11 std")
    ax1.set_ylabel("Per-Token Energy (mJ)", fontsize=12)
    title = "Stream Token Energy: Per-Position Average (decode only)" if is_stream \
        else "Batch Token Energy: Per-Position Average\n(Step energy / active requests)"
    ax1.set_title(title, fontsize=14, pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    if has_step_energy:
        # 中图: step 总能耗 (整个 batch)
        ax2 = axes[1]
        ax2.plot(positions, step_energy, color="#FF5722", linewidth=1.5,
                 label="Step energy (all GPUs, mJ)")
        ax2.set_ylabel("Step Energy (mJ)", fontsize=12)
        ax2.set_title("Total Step Energy (all requests in batch)", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper left")
        count_ax = axes[2]
    else:
        count_ax = axes[1]

    # 下图: count / active 请求数
    count_ax.fill_between(positions, 0, counts_or_active, color="#4CAF50",
                          alpha=0.4, step="mid")
    count_ax.plot(positions, counts_or_active, color="#4CAF50", linewidth=1.5,
                  label=count_legend)
    count_ax.set_xlabel("Token Position (decode step)", fontsize=12)
    count_ax.set_ylabel(count_label, fontsize=12)
    count_ax.set_title(f"{count_label} per Position", fontsize=12)
    count_ax.grid(True, alpha=0.3)
    count_ax.legend(loc="upper right")

    # 底部文字描述
    fig.text(0.5, -0.02, desc, ha="center", va="top", fontsize=10,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                       edgecolor="#BDBDBD", alpha=0.9))

    plt.tight_layout()
    path = os.path.join(output_dir, "token_energy_batch_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path}")


def plot_batch_step_vs_token_energy(pos_data, output_dir):
    """
    Batch 模式: 对比 step 总能耗 vs per-token 能耗, 展示 batch 效率。
    """
    if not pos_data:
        return

    positions = np.array([int(r["position"]) for r in pos_data])
    means = np.array([float(r["mean_energy_mj"]) for r in pos_data])

    has_active = "active_requests" in pos_data[0]
    has_step_energy = "step_energy_mj" in pos_data[0]

    if has_active:
        counts_or_active = np.array([int(r["active_requests"]) for r in pos_data])
    else:
        counts_or_active = np.array([int(r.get("count", 1)) for r in pos_data])

    step_energy = np.array([float(r["step_energy_mj"]) for r in pos_data]) \
        if has_step_energy else None

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # 主轴: per-token 能耗
    ax1.plot(positions, means, color="#2196F3", linewidth=2,
             label="Per-token energy (mJ)", alpha=0.8)

    if step_energy is not None and np.any(step_energy > 0):
        # Batch 模式: 同时画 step energy
        ax1.plot(positions, step_energy, color="#FF5722", linewidth=2,
                 label="Step energy (total batch)", alpha=0.8)
        # 标注 batch 效率
        if len(positions) > 10:
            mid = len(positions) // 2
            efficiency = means[mid] / step_energy[mid] * 100 \
                if step_energy[mid] > 0 else 0
            ax1.annotate(
                f"Batch efficiency: {counts_or_active[mid]} requests\n"
                f"Per-token = {efficiency:.1f}% of step energy",
                xy=(positions[mid], means[mid]),
                xytext=(50, 50), textcoords="offset points",
                fontsize=10, color="#2196F3",
                arrowprops=dict(arrowstyle="->", color="#2196F3"))
        ax1.set_ylabel("Energy (mJ)", fontsize=12)
        ax1.set_title("Step Energy vs Per-Token Energy (Batch Amortization)",
                       fontsize=14, pad=15)
    else:
        # Stream 模式: 用次轴画 sample count
        ax1.set_ylabel("Per-Token Energy (mJ)", fontsize=12, color="#2196F3")
        ax1.tick_params(axis='y', labelcolor="#2196F3")

        ax2 = ax1.twinx()
        ax2.fill_between(positions, 0, counts_or_active, color="#FF9800",
                         alpha=0.12, step="mid")
        ax2.plot(positions, counts_or_active, color="#FF9800", linewidth=1.5,
                 alpha=0.7, label="Sample count")
        ax2.set_ylabel("Sample Count", fontsize=12, color="#FF9800")
        ax2.tick_params(axis='y', labelcolor="#FF9800")

        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        ax1.set_title("Per-Token Energy vs Sample Count by Position",
                       fontsize=14, pad=15)

    ax1.set_xlabel("Token Position", fontsize=12)
    if step_energy is not None:
        ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 底部文字描述
    desc_lines = []
    avg_token = np.mean(means)
    if step_energy is not None and np.any(step_energy > 0):
        avg_step = np.mean(step_energy)
        avg_active = np.mean(counts_or_active)
        gap_ratio = avg_token / avg_step * 100 if avg_step > 0 else 0
        desc_lines.append(
            f"Red: GPU energy per engine step (all requests). "
            f"Mean = {avg_step:.0f} mJ/step")
        desc_lines.append(
            f"Blue: energy amortized per token = step_energy / active_requests. "
            f"Mean = {avg_token:.0f} mJ/token")
        desc_lines.append(
            f"Avg batch size = {avg_active:.0f} requests -> "
            f"per-token is ~{gap_ratio:.1f}% of step energy. "
            f"Larger batch = lower per-token cost.")
    else:
        desc_lines.append(
            f"Blue: per-token decode energy averaged across all requests "
            f"at each position. Mean = {avg_token:.0f} mJ/token")
        desc_lines.append(
            f"Orange: number of request samples contributing to each position. "
            f"Early positions have more samples (all requests pass through).")
        if len(counts_or_active) > 0:
            desc_lines.append(
                f"Count range: {int(counts_or_active[0])} (pos 1) -> "
                f"{int(counts_or_active[-1])} (pos {int(positions[-1])}). "
                f"Fewer samples at later positions = higher variance.")
    desc = "\n".join(desc_lines)

    fig.text(0.5, -0.02, desc, ha="center", va="top", fontsize=10,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                       edgecolor="#BDBDBD", alpha=0.9))

    plt.tight_layout()
    path = os.path.join(output_dir, "token_energy_batch_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path}")


def print_batch_summary(pos_data, step_raw_data):
    """打印 batch/stream 模式的汇总统计表。"""
    if not pos_data:
        print("  ⚠ No position data available.")
        return

    means = [float(r["mean_energy_mj"]) for r in pos_data]
    has_active = "active_requests" in pos_data[0]
    has_step_energy = "step_energy_mj" in pos_data[0]

    counts = [int(r.get("active_requests" if has_active else "count", 1))
              for r in pos_data]
    step_energies = [float(r["step_energy_mj"]) for r in pos_data] \
        if has_step_energy else []

    total_positions = len(pos_data)
    overall_mean = np.mean(means)
    overall_min = min(means) if means else 0
    overall_max = max(means) if means else 0

    mode_label = "BATCH" if has_active else "STREAM"

    print(f"\n{'='*60}")
    print(f"  {mode_label} TOKEN-LEVEL ENERGY SUMMARY")
    print(f"{'='*60}")
    print(f"  Total positions:          {total_positions}")
    count_label = "batch size" if has_active else "sample count"
    print(f"  Initial {count_label}:     {counts[0] if counts else 0}")
    print(f"  Final {count_label}:       {counts[-1] if counts else 0}")
    print(f"  Mean per-token energy:    {overall_mean:.2f} mJ")
    print(f"  Min per-token energy:     {overall_min:.2f} mJ")
    print(f"  Max per-token energy:     {overall_max:.2f} mJ")

    if step_energies:
        mean_step = np.mean(step_energies)
        cumulative = sum(step_energies)
        print(f"  Mean step energy:         {mean_step:.2f} mJ")
        print(f"  Total step energy:        {cumulative:.2f} mJ ({cumulative/1000:.4f} J)")

    # 分段统计
    if total_positions >= 20:
        first_10 = np.mean(means[:10])
        last_10 = np.mean(means[-10:])
        mid_start = total_positions // 2 - 5
        mid_10 = np.mean(means[mid_start:mid_start+10])

        print(f"\n  Position-wise per-token energy breakdown:")
        print(f"    First 10 tokens:      {first_10:.2f} mJ/token "
              f"({count_label}={counts[0]})")
        print(f"    Middle 10 tokens:     {mid_10:.2f} mJ/token "
              f"({count_label}={counts[total_positions//2]})")
        print(f"    Last 10 tokens:       {last_10:.2f} mJ/token "
              f"({count_label}={counts[-10]})")
        ratio = last_10 / first_10 if first_10 > 0 else 0
        print(f"    Last/First ratio:     {ratio:.2f}x")

        if step_energies:
            first_10_step = np.mean(step_energies[:10])
            last_10_step = np.mean(step_energies[-10:])
            mid_10_step = np.mean(step_energies[mid_start:mid_start+10])
            print(f"\n  Position-wise step energy breakdown:")
            print(f"    First 10 steps:       {first_10_step:.2f} mJ/step")
            print(f"    Middle 10 steps:      {mid_10_step:.2f} mJ/step")
            print(f"    Last 10 steps:        {last_10_step:.2f} mJ/step")
            step_ratio = last_10_step / first_10_step if first_10_step > 0 else 0
            print(f"    Last/First ratio:     {step_ratio:.2f}x")

    print(f"{'='*60}")


def run_visualization(data_dir=None, output_dir=None, mode="auto"):
    """
    可供外部调用的可视化入口函数。

    Args:
        data_dir:   CSV 数据所在目录 (默认: <project_root>/output)
        output_dir: 图表输出目录 (默认: <project_root>/figures)
        mode:       "auto" | "sequential" | "batch" | "stream"
    Returns:
        bool: 是否成功生成图表
    """
    _project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    if data_dir is None:
        data_dir = os.path.join(_project_root, "output")
    if output_dir is None:
        output_dir = os.path.join(_project_root, "figures")
    os.makedirs(output_dir, exist_ok=True)

    # 自动检测模式
    stream_pos_csv = os.path.join(data_dir, "token_energy_stream_per_position.csv")
    batch_pos_csv = os.path.join(data_dir, "token_energy_batch_per_position.csv")
    seq_pos_csv = os.path.join(data_dir, "token_energy_per_position.csv")

    if mode == "auto":
        if os.path.exists(stream_pos_csv):
            mode = "stream"
        elif os.path.exists(batch_pos_csv):
            mode = "batch"
        elif os.path.exists(seq_pos_csv):
            mode = "sequential"
        else:
            print(f"⚠ No token energy data found in {data_dir}, skipping visualization.")
            return False

    if mode == "stream":
        # ---- Stream 模式 (与 batch 共享可视化函数, CSV 格式兼容) ----
        pos_data = read_csv(stream_pos_csv)
        sample_csv = os.path.join(data_dir, "token_energy_stream_per_sample.csv")
        sample_data = read_csv(sample_csv)

        if not pos_data:
            print(f"⚠ No data in {stream_pos_csv}, skipping visualization.")
            return False

        print(f"\n📊 Generating STREAM token energy visualizations...")
        print(f"   Data dir:   {data_dir}")
        print(f"   Output dir: {output_dir}")
        print(f"   Positions:  {len(pos_data)}")
        print(f"   Samples:    {len(sample_data)}")
        print()

        plot_batch_energy_curve(pos_data, output_dir)
        plot_batch_step_vs_token_energy(pos_data, output_dir)
        plot_cumulative_energy(pos_data, output_dir)
        plot_energy_distribution(pos_data, output_dir)
        print_batch_summary(pos_data, [])

    elif mode == "batch":
        # ---- Batch 模式 ----
        pos_data = read_csv(batch_pos_csv)
        step_raw_csv = os.path.join(data_dir, "token_energy_batch_step_raw.csv")
        step_raw_data = read_csv(step_raw_csv)
        sample_csv = os.path.join(data_dir, "token_energy_batch_per_sample.csv")
        sample_data = read_csv(sample_csv)

        if not pos_data:
            print(f"⚠ No data in {batch_pos_csv}, skipping visualization.")
            return False

        print(f"\n📊 Generating BATCH token energy visualizations...")
        print(f"   Data dir:   {data_dir}")
        print(f"   Output dir: {output_dir}")
        print(f"   Positions:  {len(pos_data)}")
        print(f"   Step raw:   {len(step_raw_data)}")
        print(f"   Samples:    {len(sample_data)}")
        print()

        plot_batch_energy_curve(pos_data, output_dir)
        plot_batch_step_vs_token_energy(pos_data, output_dir)
        plot_cumulative_energy(pos_data, output_dir)
        plot_energy_distribution(pos_data, output_dir)
        print_batch_summary(pos_data, step_raw_data)

    else:
        # ---- Sequential 模式 (原有逻辑) ----
        pos_data = read_csv(seq_pos_csv)
        sample_csv = os.path.join(data_dir, "token_energy_per_sample.csv")
        raw_csv = os.path.join(data_dir, "token_energy_raw.csv")
        sample_data = read_csv(sample_csv)
        raw_data = read_csv(raw_csv)

        if not pos_data:
            print(f"⚠ No data in {seq_pos_csv}, skipping visualization.")
            return False

        print(f"\n📊 Generating SEQUENTIAL token energy visualizations...")
        print(f"   Data dir:   {data_dir}")
        print(f"   Output dir: {output_dir}")
        print(f"   Positions:  {len(pos_data)}")
        print(f"   Samples:    {len(sample_data)}")
        print(f"   Raw points: {len(raw_data)}")
        print()

        plot_mean_energy_curve(pos_data, output_dir)
        plot_energy_heatmap(raw_data, output_dir)
        plot_cumulative_energy(pos_data, output_dir)
        plot_sample_comparison(sample_data, output_dir)
        plot_energy_distribution(pos_data, output_dir)
        print_summary_table(pos_data, sample_data)

    print(f"\n✅ All figures saved to {output_dir}/")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Visualize per-token energy benchmark results")
    _project_root = os.path.join(os.path.dirname(__file__), "..")
    parser.add_argument("--experiment_dir", type=str, default=None,
                        help="Path to an experiment output directory "
                             "(e.g. output/token_energy_batch_Qwen3-32B_vllm_..._20260311_143052). "
                             "If provided, data_dir = experiment_dir, "
                             "output_dir = experiment_dir/figures.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing token_energy_*.csv files "
                             "(default: <project_root>/output)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save output figures "
                             "(default: <project_root>/figures)")
    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "sequential", "batch", "stream"],
                        help="Visualization mode: auto (detect), "
                             "sequential (token_energy), batch (token_energy_batch), "
                             "stream (token_energy_stream)")
    args = parser.parse_args()

    # 如果指定了 experiment_dir, 自动推断 data_dir 和 output_dir
    if args.experiment_dir:
        data_dir = args.experiment_dir
        output_dir = os.path.join(args.experiment_dir, "figures")
    else:
        data_dir = args.data_dir or os.path.join(_project_root, "output")
        output_dir = args.output_dir or os.path.join(_project_root, "figures")

    success = run_visualization(
        data_dir=data_dir,
        output_dir=output_dir,
        mode=args.mode,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
