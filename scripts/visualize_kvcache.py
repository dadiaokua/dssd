#!/usr/bin/env python3
"""
可视化 KV Cache 基准测试结果
从 results_real_network_kvcache_summary.csv 读取数据，生成图表。
"""

import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # 无头模式，直接保存文件
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── 配置 ──────────────────────────────────────────────────
_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
CSV_PATH = os.path.join(_PROJECT_ROOT, "output",
                        "results_real_network_kvcache_summary.csv")
OUT_DIR  = os.path.join(_PROJECT_ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# 需要提取的列（_mean / _std）
COLS = {
    "total_energy":   ("energy_total_mj_mean",         "energy_total_mj_std"),
    "compute_energy": ("energy_compute_mj_mean",       "energy_compute_mj_std"),
    "memory_energy":  ("energy_memory_mj_mean",        "energy_memory_mj_std"),
    "idle_energy":    ("energy_idle_mj_mean",           "energy_idle_mj_std"),
    "throughput":     ("throughput_mean",               "throughput_std"),
    "wall_time":      ("wall_time_mean",                "wall_time_std"),
    "avg_power":      ("energy_est_avg_power_mw_mean",  "energy_est_avg_power_mw_std"),
    "bytes_per_tok":  ("energy_total_bytes_per_tok_mean","energy_total_bytes_per_tok_std"),
    "kv_read":        ("energy_kv_read_bytes_per_tok_mean", "energy_kv_read_bytes_per_tok_std"),
}


def load_csv(path):
    """读取 CSV，按 (method, kv_cache_len) 聚合（取 trial 平均）"""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    # 按 trial 分组：同一 method + kv_cache_len 可能有多个 trial
    # summary 文件已经做了聚合，每行就是一个 (method, kv_cache_len, trial)
    # 我们再按 (method, kv_cache_len) 做平均
    grouped = defaultdict(list)
    for r in rows:
        key = (r["method"], int(r["kv_cache_len"]))
        grouped[key].append(r)

    result = {}
    for key, rs in grouped.items():
        agg = {}
        for name, (mean_col, std_col) in COLS.items():
            vals = [float(r[mean_col]) for r in rs if r.get(mean_col)]
            stds = [float(r[std_col]) for r in rs if r.get(std_col)]
            agg[name + "_mean"] = np.mean(vals) if vals else 0
            agg[name + "_std"]  = np.mean(stds) if stds else 0  # 平均 std 作为误差
        result[key] = agg
    return result


def make_plots(data):
    """生成所有可视化图表"""
    # 提取所有 method 和 kv_cache_len
    methods = sorted(set(k[0] for k in data))
    kv_lens = sorted(set(k[1] for k in data))

    # 颜色方案
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(methods), 3)))

    # ── 全局样式 ──
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "#f8f8f8",
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "font.size":        11,
    })

    # ════════════════════════════════════════════════════════
    # 图1: 能耗随 KV Cache 长度变化（总 / 计算 / 内存 / idle）
    # ════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Energy vs KV Cache Length", fontsize=16, fontweight="bold")

    energy_keys = [
        ("total_energy",   "Total Energy (mJ)"),
        ("compute_energy", "Compute Energy (mJ)"),
        ("memory_energy",  "Memory Energy (mJ)"),
        ("idle_energy",    "Idle Energy (mJ)"),
    ]

    for ax, (ekey, title) in zip(axes.flat, energy_keys):
        for i, method in enumerate(methods):
            means = [data.get((method, kv), {}).get(ekey + "_mean", 0) for kv in kv_lens]
            stds  = [data.get((method, kv), {}).get(ekey + "_std", 0) for kv in kv_lens]
            ax.errorbar(kv_lens, means, yerr=stds,
                        marker="o", capsize=4, linewidth=2,
                        color=colors[i], label=method)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("KV Cache Length (tokens)")
        ax.set_ylabel("Energy (mJ)")
        ax.legend(fontsize=9)
        ax.set_xticks(kv_lens)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path1 = os.path.join(OUT_DIR, "energy_vs_kvcache.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] 已保存: {path1}")

    # ════════════════════════════════════════════════════════
    # 图2: 能耗分解堆叠柱状图
    # ════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Energy Breakdown by KV Cache Length", fontsize=16, fontweight="bold")

    bar_width = 0.25
    n_methods = len(methods)
    x = np.arange(len(kv_lens))

    stack_colors = ["#e74c3c", "#3498db", "#2ecc71"]  # compute, memory, idle
    stack_labels = ["Compute", "Memory", "Idle"]
    stack_keys   = ["compute_energy", "memory_energy", "idle_energy"]

    for i, method in enumerate(methods):
        bottoms = np.zeros(len(kv_lens))
        for j, (skey, slabel, scolor) in enumerate(zip(stack_keys, stack_labels, stack_colors)):
            vals = [data.get((method, kv), {}).get(skey + "_mean", 0) for kv in kv_lens]
            label = f"{method} - {slabel}" if i == 0 else f"_{method} - {slabel}"
            ax.bar(x + i * bar_width, vals, bar_width,
                   bottom=bottoms, color=scolor, alpha=0.7 + 0.15 * i,
                   label=slabel if i == 0 else None,
                   edgecolor="white", linewidth=0.5)
            bottoms += np.array(vals)

    ax.set_xlabel("KV Cache Length (tokens)", fontsize=12)
    ax.set_ylabel("Energy (mJ)", fontsize=12)
    ax.set_xticks(x + bar_width * (n_methods - 1) / 2)
    ax.set_xticklabels([str(k) for k in kv_lens])
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}K" if v >= 1e3 else f"{v:.0f}"))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path2 = os.path.join(OUT_DIR, "energy_breakdown_stacked.png")
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] 已保存: {path2}")

    # ════════════════════════════════════════════════════════
    # 图3: 吞吐率 & 耗时
    # ════════════════════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Performance vs KV Cache Length", fontsize=16, fontweight="bold")

    for i, method in enumerate(methods):
        tp_means = [data.get((method, kv), {}).get("throughput_mean", 0) for kv in kv_lens]
        tp_stds  = [data.get((method, kv), {}).get("throughput_std", 0) for kv in kv_lens]
        ax1.errorbar(kv_lens, tp_means, yerr=tp_stds,
                     marker="s", capsize=4, linewidth=2,
                     color=colors[i], label=method)

    ax1.set_title("Throughput (tokens/s)", fontsize=13)
    ax1.set_xlabel("KV Cache Length (tokens)")
    ax1.set_ylabel("Tokens/s")
    ax1.legend(fontsize=9)
    ax1.set_xticks(kv_lens)

    for i, method in enumerate(methods):
        wt_means = [data.get((method, kv), {}).get("wall_time_mean", 0) for kv in kv_lens]
        wt_stds  = [data.get((method, kv), {}).get("wall_time_std", 0) for kv in kv_lens]
        ax2.errorbar(kv_lens, wt_means, yerr=wt_stds,
                     marker="^", capsize=4, linewidth=2,
                     color=colors[i], label=method)

    ax2.set_title("Wall Time (s)", fontsize=13)
    ax2.set_xlabel("KV Cache Length (tokens)")
    ax2.set_ylabel("Time (s)")
    ax2.legend(fontsize=9)
    ax2.set_xticks(kv_lens)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path3 = os.path.join(OUT_DIR, "performance_vs_kvcache.png")
    fig.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] 已保存: {path3}")

    # ════════════════════════════════════════════════════════
    # 图4: Memory Bytes/token 分解（KV read 随 seq_len 增长）
    # ════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Memory Bytes per Token vs KV Cache Length", fontsize=16, fontweight="bold")

    for i, method in enumerate(methods):
        total_bpt = [data.get((method, kv), {}).get("bytes_per_tok_mean", 0) / 1e9 for kv in kv_lens]
        kv_bpt    = [data.get((method, kv), {}).get("kv_read_mean", 0) / 1e6 for kv in kv_lens]

        ax.plot(kv_lens, total_bpt, marker="o", linewidth=2,
                color=colors[i], label=f"{method} - Total Bytes/tok (GB)")
        ax.plot(kv_lens, kv_bpt, marker="x", linewidth=2, linestyle="--",
                color=colors[i], alpha=0.6, label=f"{method} - KV Read/tok (MB)")

    ax.set_xlabel("KV Cache Length (tokens)")
    ax.set_ylabel("Bytes per Token")
    ax.legend(fontsize=9)
    ax.set_xticks(kv_lens)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path4 = os.path.join(OUT_DIR, "memory_bytes_per_token.png")
    fig.savefig(path4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] 已保存: {path4}")

    # ════════════════════════════════════════════════════════
    # 图5: 能耗占比饼图（每个 KV cache 长度一个）
    # ════════════════════════════════════════════════════════
    n_kv = len(kv_lens)
    fig, axes = plt.subplots(1, n_kv, figsize=(5 * n_kv, 5))
    if n_kv == 1:
        axes = [axes]
    fig.suptitle("Energy Composition per KV Cache Length", fontsize=16, fontweight="bold")

    for ax, kv in zip(axes, kv_lens):
        # 用第一个 method 的数据
        method = methods[0]
        d = data.get((method, kv), {})
        vals = [
            d.get("compute_energy_mean", 0),
            d.get("memory_energy_mean", 0),
            d.get("idle_energy_mean", 0),
        ]
        labels = ["Compute", "Memory", "Idle"]
        pie_colors = ["#e74c3c", "#3498db", "#2ecc71"]

        # 过滤掉 0 值
        filtered = [(v, l, c) for v, l, c in zip(vals, labels, pie_colors) if v > 0]
        if filtered:
            fv, fl, fc = zip(*filtered)
            wedges, texts, autotexts = ax.pie(
                fv, labels=fl, colors=fc, autopct="%1.1f%%",
                startangle=90, pctdistance=0.75)
            for t in autotexts:
                t.set_fontsize(10)
                t.set_fontweight("bold")
        ax.set_title(f"KV={kv}  ({method})", fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path5 = os.path.join(OUT_DIR, "energy_composition_pie.png")
    fig.savefig(path5, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] 已保存: {path5}")

    # ════════════════════════════════════════════════════════
    # 图6: 综合数据表格
    # ════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(16, 2 + 0.4 * len(kv_lens) * len(methods)))
    ax.axis("off")
    fig.suptitle("Summary Table", fontsize=16, fontweight="bold")

    headers = ["Method", "KV Len", "Total (mJ)", "Compute (mJ)", "Memory (mJ)",
               "Idle (mJ)", "Throughput", "Wall Time (s)"]
    table_data = []
    for method in methods:
        for kv in kv_lens:
            d = data.get((method, kv), {})
            table_data.append([
                method, str(kv),
                f"{d.get('total_energy_mean', 0):,.0f}",
                f"{d.get('compute_energy_mean', 0):,.0f}",
                f"{d.get('memory_energy_mean', 0):,.0f}",
                f"{d.get('idle_energy_mean', 0):,.0f}",
                f"{d.get('throughput_mean', 0):.2f} tok/s",
                f"{d.get('wall_time_mean', 0):.2f}",
            ])

    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(list(range(len(headers))))

    # 表头样式
    for j in range(len(headers)):
        table[0, j].set_facecolor("#34495e")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # 交替行颜色
    for i in range(len(table_data)):
        color = "#ecf0f1" if i % 2 == 0 else "white"
        for j in range(len(headers)):
            table[i + 1, j].set_facecolor(color)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path6 = os.path.join(OUT_DIR, "summary_table.png")
    fig.savefig(path6, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] 已保存: {path6}")

    print(f"\n[完成] 所有图表已保存到 {OUT_DIR}/")
    print(f"  共 6 张图：")
    print(f"  1. energy_vs_kvcache.png        - 能耗随 KV Cache 长度变化")
    print(f"  2. energy_breakdown_stacked.png  - 能耗分解堆叠柱状图")
    print(f"  3. performance_vs_kvcache.png    - 吞吐率 & 耗时")
    print(f"  4. memory_bytes_per_token.png    - 每 token 内存搬运量")
    print(f"  5. energy_composition_pie.png    - 能耗占比饼图")
    print(f"  6. summary_table.png             - 综合数据表格")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = CSV_PATH
    print(f"[读取] {csv_path}")
    data = load_csv(csv_path)
    print(f"[数据] {len(data)} 组 (method, kv_cache_len) 组合")
    make_plots(data)
