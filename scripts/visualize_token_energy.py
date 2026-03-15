#!/usr/bin/env python3
"""
visualize_token_energy.py - 逐 token 能耗可视化 (NeurIPS/ICLR style)

读取 token_energy_per_position.csv, token_energy_per_sample.csv,
token_energy_raw.csv, 生成可视化图表。

用法:
    python visualize_token_energy.py [--data_dir ./] [--output_dir ./figures]
    python visualize_token_energy.py --experiment_dir output/<exp_name>

生成图表:
  1. Per-position 平均能耗曲线 (带 IQR 置信带 + 线性拟合)
  2. Per-token energy vs batch/sample count 对比
  3. 累积能耗曲线
  4. 能耗分布直方图
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
    from matplotlib.ticker import AutoMinorLocator
except ImportError:
    print("ERROR: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════
#  Global Style — NeurIPS / ICLR paper figure conventions
# ══════════════════════════════════════════════════════════════

# Academic color palette (colorblind-friendly, muted tones)
_COLORS = {
    "blue":    "#4878D0",   # primary
    "orange":  "#EE854A",   # secondary
    "green":   "#6ACC64",   # tertiary
    "red":     "#D65F5F",   # accent / fit line
    "purple":  "#956CB4",
    "gray":    "#8C8C8C",
    "pink":    "#DC7EC0",
    "brown":   "#797979",
    "light_blue": "#A6CEE3",
    "light_orange": "#FDBF6F",
}

_FIG_DPI = 300
_FONT_SIZE = {
    "title": 13,
    "label": 11,
    "tick": 9.5,
    "legend": 9,
    "annotation": 8.5,
    "caption": 8.5,
}


def _apply_style():
    """Apply publication-quality matplotlib rcParams."""
    plt.rcParams.update({
        # Font
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "font.size": _FONT_SIZE["tick"],

        # Axes
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.labelsize": _FONT_SIZE["label"],
        "axes.titlesize": _FONT_SIZE["title"],
        "axes.titleweight": "bold",
        "axes.titlepad": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Grid
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "grid.alpha": 0.7,

        # Ticks
        "xtick.labelsize": _FONT_SIZE["tick"],
        "ytick.labelsize": _FONT_SIZE["tick"],
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,

        # Legend
        "legend.fontsize": _FONT_SIZE["legend"],
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
        "legend.fancybox": True,
        "legend.borderpad": 0.4,

        # Figure
        "figure.dpi": _FIG_DPI,
        "savefig.dpi": _FIG_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,

        # Lines
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
    })


_apply_style()


# ══════════════════════════════════════════════════════════════
#  Helper utilities
# ══════════════════════════════════════════════════════════════

def read_csv(path):
    """读取 CSV 文件, 返回 list of dict。"""
    if not os.path.exists(path):
        print(f"⚠ File not found: {path}")
        return []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _smooth(arr, window):
    """Simple moving average."""
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='valid')


def _smooth_positions(positions, window):
    """Return the position array aligned with the smoothed output."""
    n = len(positions) - window + 1
    return positions[window // 2: window // 2 + n]


def _add_caption(fig, text, y=-0.03):
    """Add a figure caption below the plot (paper-style)."""
    fig.text(0.5, y, text, ha="center", va="top",
             fontsize=_FONT_SIZE["caption"],
             fontstyle="italic", color="#555555",
             wrap=True)


def _format_energy(val):
    """Format energy value with appropriate unit."""
    if val >= 1e6:
        return f"{val/1e6:.2f} kJ"
    elif val >= 1e3:
        return f"{val/1e3:.2f} J"
    else:
        return f"{val:.1f} mJ"


def _savefig(fig, path):
    """Save figure with tight layout."""
    fig.savefig(path, dpi=_FIG_DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✅ {path}")


# ══════════════════════════════════════════════════════════════
#  Figure 1: Per-Position Energy Curve (Sequential mode)
# ══════════════════════════════════════════════════════════════

def plot_mean_energy_curve(pos_data, output_dir):
    """Sequential mode: per-position mean energy with IQR band."""
    if not pos_data:
        return

    positions = np.array([int(r["position"]) for r in pos_data])
    means = np.array([float(r["mean_energy_mj"]) for r in pos_data])
    counts = np.array([int(r["count"]) for r in pos_data])

    window = max(5, min(200, len(positions) // 80))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 4.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})

    # ---- Top: energy curve ----
    if len(means) > window * 2:
        smoothed = _smooth(means, window)
        smooth_pos = _smooth_positions(positions, window)

        from numpy.lib.stride_tricks import sliding_window_view
        windowed = sliding_window_view(means, window)
        p25 = np.percentile(windowed, 25, axis=1)
        p75 = np.percentile(windowed, 75, axis=1)

        ax1.fill_between(smooth_pos, p25, p75,
                         alpha=0.25, color=_COLORS["light_blue"],
                         edgecolor="none",
                         label="P25-P75 range (middle 50%)")
        ax1.plot(smooth_pos, smoothed, color=_COLORS["blue"], linewidth=1.8,
                 label=f"Moving avg (w={window})")
    else:
        ax1.plot(positions, means, color=_COLORS["blue"], linewidth=1.5,
                 label="Mean energy")

    ax1.set_ylabel("Energy per Token (mJ)")
    ax1.set_title("Per-Token Decode Energy by Position")
    ax1.legend(loc="upper left", fontsize=_FONT_SIZE["legend"])

    # ---- Bottom: sample count ----
    ax2.fill_between(positions, 0, counts, alpha=0.35,
                     color=_COLORS["orange"], edgecolor="none")
    ax2.plot(positions, counts, color=_COLORS["orange"], linewidth=1.0)
    ax2.set_xlabel("Token Position")
    ax2.set_ylabel("Sample Count")
    ax2.set_ylim(bottom=0)

    _savefig(fig, os.path.join(output_dir, "token_energy_mean_curve.png"))


# ══════════════════════════════════════════════════════════════
#  Figure: Energy Heatmap (Sequential mode)
# ══════════════════════════════════════════════════════════════

def plot_energy_heatmap(raw_data, output_dir):
    """Heatmap of per-token energy across sequences."""
    if not raw_data:
        return

    seq_indices = set()
    max_pos = 0
    for r in raw_data:
        seq_indices.add(int(r["sequence_idx"]))
        max_pos = max(max_pos, int(r["position"]))

    n_seqs = len(seq_indices)
    if n_seqs == 0:
        return

    matrix = np.full((n_seqs, max_pos + 1), np.nan)
    for r in raw_data:
        matrix[int(r["sequence_idx"]), int(r["position"])] = float(r["energy_mj"])

    fig, ax = plt.subplots(figsize=(7, max(2.5, n_seqs * 0.25 + 0.8)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Sequence Index")
    ax.set_title("Token Energy Heatmap (darker = higher energy)")

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Energy (mJ)", fontsize=_FONT_SIZE["label"] - 1)
    cbar.ax.tick_params(labelsize=_FONT_SIZE["tick"] - 1)

    _savefig(fig, os.path.join(output_dir, "token_energy_heatmap.png"))


# ══════════════════════════════════════════════════════════════
#  Figure: Cumulative Energy
# ══════════════════════════════════════════════════════════════

def plot_cumulative_energy(pos_data, output_dir):
    """Cumulative energy consumption vs. token position."""
    if not pos_data:
        return

    positions = np.array([int(r["position"]) for r in pos_data])
    means = np.array([float(r["mean_energy_mj"]) for r in pos_data])
    cumulative = np.cumsum(means)

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.fill_between(positions, 0, cumulative / 1000,
                    alpha=0.15, color=_COLORS["green"], edgecolor="none")
    ax.plot(positions, cumulative / 1000, color=_COLORS["green"], linewidth=1.8)

    ax.set_xlabel("Token Position")
    ax.set_ylabel("Cumulative Energy (J)")
    ax.set_title("Cumulative Decode Energy")

    # Annotate total
    total_j = cumulative[-1] / 1000 if len(cumulative) > 0 else 0
    ax.annotate(f"Total: {total_j:.2f} J",
                xy=(positions[-1], total_j),
                xytext=(-25, -15), textcoords="offset points",
                fontsize=_FONT_SIZE["annotation"],
                color=_COLORS["green"],
                ha="right",
                arrowprops=dict(arrowstyle="-", color=_COLORS["gray"],
                                lw=0.6))

    # Caption
    n_pos = len(positions)
    avg = np.mean(means)
    half = n_pos // 2
    first_half = cumulative[half - 1] / 1000 if half > 0 else 0
    second_half = total_j - first_half
    ratio = second_half / first_half if first_half > 0 else 0
    caption = (f"{n_pos} positions · avg {avg:.1f} mJ/token · "
               f"1st half {first_half:.1f} J · 2nd half {second_half:.1f} J "
               f"(ratio {ratio:.2f}x, >1 means energy grows faster later)")
    _add_caption(fig, caption)

    _savefig(fig, os.path.join(output_dir, "token_energy_cumulative.png"))


# ══════════════════════════════════════════════════════════════
#  Figure: Sample Comparison (Sequential mode)
# ══════════════════════════════════════════════════════════════

def plot_sample_comparison(sample_data, output_dir):
    """Per-sample throughput and generation time."""
    if not sample_data:
        return

    indices = list(range(len(sample_data)))
    sources = [r["source"] for r in sample_data]
    throughputs = [float(r["throughput"]) for r in sample_data]
    wall_times = [float(r["wall_time"]) for r in sample_data]

    source_colors = {
        "LongForm": _COLORS["blue"],
        "python_code": _COLORS["orange"],
        "WizardLM": _COLORS["green"],
    }
    colors = [source_colors.get(s, _COLORS["gray"]) for s in sources]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    ax1.bar(indices, throughputs, color=colors, alpha=0.85, edgecolor="white",
            linewidth=0.3)
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Throughput (tok/s)")
    ax1.set_title("Throughput per Sample")

    ax2.bar(indices, wall_times, color=colors, alpha=0.85, edgecolor="white",
            linewidth=0.3)
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("Wall Time (s)")
    ax2.set_title("Generation Time per Sample")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, alpha=0.85, label=s)
                       for s, c in source_colors.items() if s in sources]
    if legend_elements:
        ax1.legend(handles=legend_elements, loc="upper right")

    fig.tight_layout(w_pad=2)
    _savefig(fig, os.path.join(output_dir, "token_energy_sample_comparison.png"))


# ══════════════════════════════════════════════════════════════
#  Figure: Energy Distribution Histogram
# ══════════════════════════════════════════════════════════════

def plot_energy_distribution(pos_data, output_dir):
    """Histogram of per-position mean energy values with auto x-axis clipping."""
    if not pos_data:
        return

    means = np.array([float(r["mean_energy_mj"]) for r in pos_data])

    fig, ax = plt.subplots(figsize=(5, 3.2))

    # Auto-clip x-axis: use P1–P99 with 20% margin to avoid long empty tails
    p1, p99 = np.percentile(means, 1), np.percentile(means, 99)
    iqr_range = p99 - p1
    clip_lo = max(means.min(), p1 - 0.3 * iqr_range)
    clip_hi = p99 + 0.3 * iqr_range
    n_outliers_hi = int(np.sum(means > clip_hi))
    n_outliers_lo = int(np.sum(means < clip_lo))

    # Only clip if the range is significantly smaller than the raw range
    do_clip = (clip_hi - clip_lo) < 0.8 * (means.max() - means.min())

    if do_clip:
        plot_means = means[(means >= clip_lo) & (means <= clip_hi)]
    else:
        plot_means = means
        clip_lo, clip_hi = means.min(), means.max()

    n_bins = min(60, max(20, len(plot_means) // 200))
    ax.hist(plot_means, bins=n_bins, color=_COLORS["blue"], alpha=0.7,
            edgecolor="white", linewidth=0.4)

    overall_mean = np.mean(means)
    median_val = np.median(means)

    ax.axvline(overall_mean, color=_COLORS["red"], linestyle="--",
               linewidth=1.2, label=f"Mean: {overall_mean:.1f} mJ")
    ax.axvline(median_val, color=_COLORS["orange"], linestyle=":",
               linewidth=1.2, label=f"Median: {median_val:.1f} mJ")

    # Set tight x-axis limits
    margin = (clip_hi - clip_lo) * 0.05
    ax.set_xlim(clip_lo - margin, clip_hi + margin)

    ax.set_xlabel("Energy per Token (mJ)")
    ax.set_ylabel("Count (positions)")
    ax.set_title("Per-Token Energy Distribution")
    ax.legend(loc="upper right", fontsize=_FONT_SIZE["legend"])

    # Caption
    std_val = np.std(means)
    cv = std_val / overall_mean * 100 if overall_mean > 0 else 0
    outlier_note = ""
    if do_clip and (n_outliers_hi + n_outliers_lo) > 0:
        outlier_note = f" · {n_outliers_hi + n_outliers_lo} outliers clipped"
    caption = (f"N={len(means)} positions · "
               f"std={std_val:.1f} mJ · "
               f"CV={cv:.1f}% (std/mean, lower=more stable) · "
               f"range [{means.min():.0f}, {means.max():.0f}] mJ{outlier_note}")
    _add_caption(fig, caption)

    _savefig(fig, os.path.join(output_dir, "token_energy_distribution.png"))


# ══════════════════════════════════════════════════════════════
#  Figure: Main Per-Position Curve (Batch & Stream)
# ══════════════════════════════════════════════════════════════

def plot_batch_energy_curve(pos_data, output_dir):
    """
    Primary figure: per-position decode energy with IQR band,
    linear fit, and count/active subplot.
    """
    if not pos_data:
        return

    positions = np.array([int(r["position"]) for r in pos_data])
    means = np.array([float(r["mean_energy_mj"]) for r in pos_data])
    stds = np.array([float(r.get("std_energy_mj", 0)) for r in pos_data])

    has_active = "active_requests" in pos_data[0]
    has_step_energy = "step_energy_mj" in pos_data[0]
    is_stream = not has_active

    if has_active:
        counts = np.array([int(r["active_requests"]) for r in pos_data])
        count_label = "Active Requests"
    else:
        counts = np.array([int(r.get("count", 1)) for r in pos_data])
        count_label = "Sample Count"

    step_energy = np.array([float(r["step_energy_mj"]) for r in pos_data]) \
        if has_step_energy else None

    # Layout: 2 or 3 subplots
    n_panels = 3 if has_step_energy else 2
    ratios = [3, 1.8, 1] if n_panels == 3 else [3, 1]
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(7, 2.2 * n_panels + 0.5), sharex=True,
        gridspec_kw={"height_ratios": ratios, "hspace": 0.08})

    # ---- Panel 1: Per-token energy ----
    ax1 = axes[0]
    window = max(5, min(200, len(positions) // 80))

    if len(means) > window * 2:
        smoothed = _smooth(means, window)
        smooth_pos = _smooth_positions(positions, window)

        from numpy.lib.stride_tricks import sliding_window_view
        windowed = sliding_window_view(means, window)
        p25 = np.percentile(windowed, 25, axis=1)
        p75 = np.percentile(windowed, 75, axis=1)

        # P25-P75: middle 50% of data, showing typical fluctuation range
        ax1.fill_between(smooth_pos, p25, p75,
                         alpha=0.25, color=_COLORS["light_blue"],
                         edgecolor="none",
                         label="P25-P75 range (middle 50%)")
        # Moving average: smooths noise by averaging nearby points
        ax1.plot(smooth_pos, smoothed, color=_COLORS["blue"], linewidth=1.8,
                 label=f"Moving avg (w={window})")
    else:
        ax1.plot(positions, means, color=_COLORS["blue"], linewidth=1.5,
                 label="Mean energy/token")

    # Linear fit (skip first 5% unstable region)
    fit_start = max(1, int(len(positions) * 0.05))
    if len(positions) > fit_start + 10:
        fit_pos = positions[fit_start:]
        fit_means = means[fit_start:]
        coeffs = np.polyfit(fit_pos, fit_means, 1)
        fit_line = np.polyval(coeffs, fit_pos)
        # R²: goodness of fit (1.0 = perfect linear, 0 = no correlation)
        ss_res = np.sum((fit_means - fit_line) ** 2)
        ss_tot = np.sum((fit_means - np.mean(fit_means)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        slope_str = (f"{coeffs[0]:.4f}" if abs(coeffs[0]) < 1
                     else f"{coeffs[0]:.2f}")
        slope_label = f"Linear fit: slope={slope_str} mJ/pos, R²={r2:.3f}"
        ax1.plot(fit_pos, fit_line, color=_COLORS["red"], linewidth=1.2,
                 linestyle="--", alpha=0.85, label=slope_label)

    # Auto-clip y-axis to avoid outlier spikes compressing the trend
    p1_y, p99_y = np.percentile(means, 1), np.percentile(means, 99)
    y_range = p99_y - p1_y
    y_lo = max(0, p1_y - 0.5 * y_range)
    y_hi = p99_y + 0.5 * y_range
    if y_hi < means.max() * 0.85:  # only clip if outliers are significant
        ax1.set_ylim(y_lo, y_hi)

    ax1.set_ylabel("Energy/Token (mJ)")
    title = ("Per-Token Decode Energy by Position (Stream)"
             if is_stream else
             "Per-Token Decode Energy by Position (Batch)")
    ax1.set_title(title)
    ax1.legend(loc="upper left", fontsize=_FONT_SIZE["legend"])

    # ---- Panel 2 (optional): Step energy ----
    if has_step_energy and step_energy is not None:
        ax2 = axes[1]
        if len(step_energy) > window * 2:
            smooth_step = _smooth(step_energy, window)
            smooth_sp = _smooth_positions(positions, window)
            ax2.plot(smooth_sp, smooth_step, color=_COLORS["orange"],
                     linewidth=1.5)
        else:
            ax2.plot(positions, step_energy, color=_COLORS["orange"],
                     linewidth=1.2)
        ax2.set_ylabel("Step Energy (mJ)")
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        count_ax = axes[2]
    else:
        count_ax = axes[1]

    # ---- Bottom panel: count / active ----
    count_ax.fill_between(positions, 0, counts, alpha=0.3,
                          color=_COLORS["green"], edgecolor="none",
                          step="mid")
    count_ax.plot(positions, counts, color=_COLORS["green"], linewidth=1.0)
    count_ax.set_xlabel("Token Position")
    count_ax.set_ylabel(count_label)
    count_ax.set_ylim(bottom=0)

    # Caption
    early = np.mean(means[:min(50, len(means))])
    late = np.mean(means[-min(50, len(means)):])
    trend = "+" if late > early * 1.03 else "-" if late < early * 0.97 else "~"
    caption = (f"{len(positions)} positions · "
               f"mean {np.mean(means):.1f} mJ/token · "
               f"early {early:.0f} -> late {late:.0f} mJ ({trend}) · "
               f"{count_label}: {int(counts[0])}->{int(counts[-1])}")
    _add_caption(fig, caption, y=-0.04)

    _savefig(fig, os.path.join(output_dir, "token_energy_batch_curve.png"))


# ══════════════════════════════════════════════════════════════
#  Figure: Step vs Per-Token Energy Comparison
# ══════════════════════════════════════════════════════════════

def plot_batch_step_vs_token_energy(pos_data, output_dir):
    """
    Batch: step energy vs per-token energy (amortization).
    Stream: per-token energy vs sample count (dual y-axis).
    """
    if not pos_data:
        return

    positions = np.array([int(r["position"]) for r in pos_data])
    means = np.array([float(r["mean_energy_mj"]) for r in pos_data])

    has_active = "active_requests" in pos_data[0]
    has_step_energy = "step_energy_mj" in pos_data[0]

    if has_active:
        counts = np.array([int(r["active_requests"]) for r in pos_data])
    else:
        counts = np.array([int(r.get("count", 1)) for r in pos_data])

    step_energy = np.array([float(r["step_energy_mj"]) for r in pos_data]) \
        if has_step_energy else None

    window = max(5, min(200, len(positions) // 80))

    fig, ax1 = plt.subplots(figsize=(7, 3.5))

    if step_energy is not None and np.any(step_energy > 0):
        # ---- Batch mode: step energy vs per-token energy ----
        if len(means) > window * 2:
            smooth_means = _smooth(means, window)
            smooth_step = _smooth(step_energy, window)
            smooth_pos = _smooth_positions(positions, window)
            ax1.plot(smooth_pos, smooth_means, color=_COLORS["blue"],
                     linewidth=1.8, label="Per-token energy")
            ax1.plot(smooth_pos, smooth_step, color=_COLORS["orange"],
                     linewidth=1.8, label="Step energy (shared by all tokens)")
        else:
            ax1.plot(positions, means, color=_COLORS["blue"], linewidth=1.5,
                     label="Per-token energy")
            ax1.plot(positions, step_energy, color=_COLORS["orange"],
                     linewidth=1.5, label="Step energy (shared by all tokens)")

        # Annotate amortization
        if len(positions) > 10:
            mid = len(positions) // 2
            eff = means[mid] / step_energy[mid] * 100 if step_energy[mid] > 0 else 0
            ax1.annotate(
                f"batch={int(counts[mid])}, per-token = {eff:.1f}% of step",
                xy=(positions[mid], means[mid]),
                xytext=(20, 25), textcoords="offset points",
                fontsize=_FONT_SIZE["annotation"],
                color=_COLORS["blue"],
                arrowprops=dict(arrowstyle="-", color=_COLORS["gray"], lw=0.6))

        ax1.set_ylabel("Energy (mJ)")
        ax1.set_title("Step Energy vs Per-Token Energy (Batch Amortization)")
        ax1.legend(loc="upper left", fontsize=_FONT_SIZE["legend"])

    else:
        # ---- Stream mode: energy + sample count (dual y-axis) ----
        ax1.spines["right"].set_visible(True)

        if len(means) > window * 2:
            smooth_means = _smooth(means, window)
            smooth_pos = _smooth_positions(positions, window)
            ax1.plot(smooth_pos, smooth_means, color=_COLORS["blue"],
                     linewidth=1.8, label="Per-token energy")
        else:
            ax1.plot(positions, means, color=_COLORS["blue"], linewidth=1.5,
                     label="Per-token energy")

        # Auto-clip y-axis for stream mode
        p1_y, p99_y = np.percentile(means, 1), np.percentile(means, 99)
        y_range = p99_y - p1_y
        y_lo = max(0, p1_y - 0.5 * y_range)
        y_hi = p99_y + 0.5 * y_range
        if y_hi < means.max() * 0.85:
            ax1.set_ylim(y_lo, y_hi)

        ax1.set_ylabel("Energy/Token (mJ)", color=_COLORS["blue"])
        ax1.tick_params(axis='y', colors=_COLORS["blue"])

        ax2 = ax1.twinx()
        ax2.spines["right"].set_visible(True)
        ax2.spines["top"].set_visible(False)
        ax2.fill_between(positions, 0, counts, color=_COLORS["orange"],
                         alpha=0.15, edgecolor="none", step="mid")
        ax2.plot(positions, counts, color=_COLORS["orange"], linewidth=1.0,
                 alpha=0.7, label="Sample count")
        ax2.set_ylabel("Sample Count", color=_COLORS["orange"])
        ax2.tick_params(axis='y', colors=_COLORS["orange"])

        # Merge legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
                   fontsize=_FONT_SIZE["legend"])

        ax1.set_title("Per-Token Energy vs Sample Count")

    ax1.set_xlabel("Token Position")

    # Caption
    avg_tok = np.mean(means)
    if step_energy is not None and np.any(step_energy > 0):
        avg_step = np.mean(step_energy)
        avg_active = np.mean(counts)
        gap = avg_tok / avg_step * 100 if avg_step > 0 else 0
        caption = (f"avg step={avg_step:.0f} mJ · "
                   f"avg per-token={avg_tok:.0f} mJ · "
                   f"avg batch={avg_active:.0f} · "
                   f"ratio={gap:.1f}% (per-token / step)")
    else:
        caption = (f"avg per-token={avg_tok:.0f} mJ · "
                   f"count: {int(counts[0])}->{int(counts[-1])} "
                   f"(fewer samples at later positions -> higher variance)")
    _add_caption(fig, caption)

    _savefig(fig, os.path.join(output_dir, "token_energy_batch_comparison.png"))


# ══════════════════════════════════════════════════════════════
#  Summary Tables (text output)
# ══════════════════════════════════════════════════════════════

def print_summary_table(pos_data, sample_data):
    """Print summary statistics (sequential mode)."""
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
        from collections import Counter
        source_dist = Counter(r["source"] for r in sample_data)
        print(f"\n  Samples:                {n_samples}")
        print(f"  Avg throughput:         {avg_throughput:.2f} tokens/s")
        print(f"  Avg wall time:          {avg_wall_time:.2f} s")
        print(f"  Dataset distribution:   {dict(source_dist)}")

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


def print_batch_summary(pos_data, step_raw_data):
    """Print batch/stream mode summary statistics."""
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
    count_label = "batch size" if has_active else "sample count"

    print(f"\n{'='*60}")
    print(f"  {mode_label} TOKEN-LEVEL ENERGY SUMMARY")
    print(f"{'='*60}")
    print(f"  Total positions:          {total_positions}")
    print(f"  Initial {count_label}:     {counts[0] if counts else 0}")
    print(f"  Final {count_label}:       {counts[-1] if counts else 0}")
    print(f"  Mean per-token energy:    {overall_mean:.2f} mJ")
    print(f"  Min per-token energy:     {overall_min:.2f} mJ")
    print(f"  Max per-token energy:     {overall_max:.2f} mJ")

    if step_energies:
        mean_step = np.mean(step_energies)
        cumulative = sum(step_energies)
        print(f"  Mean step energy:         {mean_step:.2f} mJ")
        print(f"  Total step energy:        {cumulative:.2f} mJ "
              f"({cumulative/1000:.4f} J)")

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


# ══════════════════════════════════════════════════════════════
#  Figure: TTFT & Latency Distribution
# ══════════════════════════════════════════════════════════════

def plot_ttft_latency(sample_data, output_dir):
    """
    Generate TTFT (Time To First Token) and per-request latency distribution.

    Reads the per-sample CSV which now includes ttft_s and latency_s columns.
    Produces a 2-row figure: TTFT histogram + Latency histogram.
    """
    _apply_style()
    if not sample_data:
        return

    # Filter non-warmup requests with valid TTFT/Latency
    ttfts = []
    latencies = []
    gen_counts = []
    for row in sample_data:
        if str(row.get("is_warmup", "")).lower() in ("true", "1"):
            continue
        t = row.get("ttft_s", "")
        l = row.get("latency_s", "")
        if t and t != "":
            try:
                ttfts.append(float(t))
            except (ValueError, TypeError):
                pass
        if l and l != "":
            try:
                latencies.append(float(l))
            except (ValueError, TypeError):
                pass
        gc = row.get("generated_tokens", 0)
        try:
            gen_counts.append(int(gc))
        except (ValueError, TypeError):
            pass

    if not ttfts and not latencies:
        print("  ⚠ No TTFT/Latency data found, skipping ttft_latency plot.")
        return

    n_plots = sum([bool(ttfts), bool(latencies)])
    fig, axes = plt.subplots(n_plots, 1, figsize=(7, 3.5 * n_plots))
    if n_plots == 1:
        axes = [axes]
    ax_idx = 0

    if ttfts:
        ax = axes[ax_idx]
        ax_idx += 1
        ttft_arr = np.array(ttfts)
        n_bins = min(50, max(15, len(ttft_arr) // 5))
        ax.hist(ttft_arr, bins=n_bins, color="#2196F3", alpha=0.85,
                edgecolor="white", linewidth=0.5)
        ax.axvline(np.mean(ttft_arr), color="#FF5722", linestyle="--",
                   linewidth=1.5, label=f"Mean = {np.mean(ttft_arr):.3f}s")
        ax.axvline(np.median(ttft_arr), color="#4CAF50", linestyle=":",
                   linewidth=1.5, label=f"Median = {np.median(ttft_arr):.3f}s")
        if len(ttft_arr) >= 20:
            p95 = np.percentile(ttft_arr, 95)
            ax.axvline(p95, color="#9C27B0", linestyle="-.",
                       linewidth=1.2, label=f"P95 = {p95:.3f}s")
        ax.set_xlabel("TTFT — Time To First Token (seconds)")
        ax.set_ylabel("Count")
        ax.set_title("First Token Latency Distribution")
        ax.legend(fontsize=8)

        # Finished/total stats
        finished = sum(1 for r in sample_data
                       if str(r.get("is_warmup", "")).lower() not in ("true", "1")
                       and str(r.get("finished", "")).lower() in ("true", "1"))
        total = sum(1 for r in sample_data
                    if str(r.get("is_warmup", "")).lower() not in ("true", "1"))
        caption = (f"N={len(ttfts)} requests | Completed: {finished}/{total} "
                   f"({finished/max(1,total)*100:.0f}%) | "
                   f"Min={min(ttfts):.3f}s, Max={max(ttfts):.3f}s")
        ax.text(0.5, -0.18, caption, transform=ax.transAxes,
                fontsize=7, ha="center", style="italic", color="#555")

    if latencies:
        ax = axes[ax_idx]
        ax_idx += 1
        lat_arr = np.array(latencies)
        n_bins = min(50, max(15, len(lat_arr) // 5))
        ax.hist(lat_arr, bins=n_bins, color="#FF9800", alpha=0.85,
                edgecolor="white", linewidth=0.5)
        ax.axvline(np.mean(lat_arr), color="#FF5722", linestyle="--",
                   linewidth=1.5, label=f"Mean = {np.mean(lat_arr):.1f}s")
        ax.axvline(np.median(lat_arr), color="#4CAF50", linestyle=":",
                   linewidth=1.5, label=f"Median = {np.median(lat_arr):.1f}s")
        if len(lat_arr) >= 20:
            p95 = np.percentile(lat_arr, 95)
            ax.axvline(p95, color="#9C27B0", linestyle="-.",
                       linewidth=1.2, label=f"P95 = {p95:.1f}s")
        ax.set_xlabel("End-to-End Latency (seconds)")
        ax.set_ylabel("Count")
        ax.set_title("Request Completion Latency Distribution")
        ax.legend(fontsize=8)

        avg_gen = np.mean(gen_counts) if gen_counts else 0
        caption = (f"N={len(latencies)} completed requests | "
                   f"Avg generated tokens: {avg_gen:.0f} | "
                   f"Min={min(latencies):.1f}s, Max={max(latencies):.1f}s")
        ax.text(0.5, -0.18, caption, transform=ax.transAxes,
                fontsize=7, ha="center", style="italic", color="#555")

    fig.tight_layout(h_pad=3.0)
    out_path = os.path.join(output_dir, "ttft_latency_distribution.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ══════════════════════════════════════════════════════════════
#  Figure: Position-Aware Token Cost Model
# ══════════════════════════════════════════════════════════════

def plot_token_cost_model(pos_data, output_dir):
    """
    Generate a position-aware token cost analysis figure.

    Core insight: later tokens cost more energy due to growing KV cache,
    but uniform pricing charges the same per token. This function fits a
    linear energy model  E(p) = alpha + beta * p  and derives a
    position-aware cost weight  w(p) = E(p) / E_mean  so that the total
    cost of a sequence is preserved while each token pays its fair share.

    Produces a 3-panel figure:
      Panel 1: Actual energy curve + uniform pricing line + linear model
      Panel 2: Position-aware cost weight w(p)
      Panel 3: Cumulative cost comparison (uniform vs position-aware)
    """
    if not pos_data or len(pos_data) < 20:
        return

    positions = np.array([int(r["position"]) for r in pos_data])
    means = np.array([float(r["mean_energy_mj"]) for r in pos_data])

    # Skip position 0/1 which may have prefill residual
    mask = positions >= 2
    if mask.sum() < 20:
        return
    pos = positions[mask].astype(float)
    eng = means[mask]

    n = len(pos)
    overall_mean = eng.mean()

    # ---- Fit linear model: E(p) = alpha + beta * p ----
    coeffs = np.polyfit(pos, eng, 1)
    beta, alpha = coeffs[0], coeffs[1]
    model_line = np.polyval(coeffs, pos)

    # R²
    ss_res = np.sum((eng - model_line) ** 2)
    ss_tot = np.sum((eng - overall_mean) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # ---- Position-aware cost weight: w(p) = E_model(p) / E_mean ----
    # This ensures: sum(w(p)) / N = 1, so total cost is unchanged
    weights = model_line / overall_mean

    # ---- Quartile fairness analysis ----
    q_size = n // 4
    q_labels = ["Q1\n(early)", "Q2", "Q3", "Q4\n(late)"]
    q_actual_frac = []
    q_uniform_frac = []
    q_model_frac = []
    for i in range(4):
        s, e = i * q_size, (i + 1) * q_size if i < 3 else n
        q_eng = eng[s:e].sum()
        q_mod = model_line[s:e].sum()
        total_eng = eng.sum()
        total_mod = model_line.sum()
        q_actual_frac.append(q_eng / total_eng * 100)
        q_uniform_frac.append((e - s) / n * 100)
        q_model_frac.append(q_mod / total_mod * 100)

    # ---- Smoothing for display ----
    window = max(5, min(200, n // 80))

    # ═══════════════ Create figure ═══════════════
    fig, axes = plt.subplots(
        3, 1, figsize=(7, 8.5), sharex=False,
        gridspec_kw={"height_ratios": [2.5, 1.5, 2], "hspace": 0.35})

    # ---- Panel 1: Energy curve + uniform + linear model ----
    ax1 = axes[0]

    # Smoothed actual energy
    if n > window * 2:
        smooth_eng = _smooth(eng, window)
        smooth_pos = _smooth_positions(pos, window)
        ax1.plot(smooth_pos, smooth_eng, color=_COLORS["blue"],
                 linewidth=1.5, alpha=0.7, label="Measured (moving avg)")
    else:
        ax1.plot(pos, eng, color=_COLORS["blue"], linewidth=1.2,
                 alpha=0.7, label="Measured energy")

    # Uniform pricing line (flat)
    ax1.axhline(overall_mean, color=_COLORS["gray"], linewidth=1.5,
                linestyle=":", label=f"Uniform pricing = {overall_mean:.0f} mJ",
                zorder=3)

    # Linear model
    ax1.plot(pos, model_line, color=_COLORS["red"], linewidth=2.0,
             linestyle="--",
             label=(f"Linear model: E(p) = {alpha:.0f} + {beta:.4f}p  "
                    f"(R\u00b2={r2:.3f})"),
             zorder=4)

    # Shade the "overpay" and "underpay" regions
    ax1.fill_between(pos, overall_mean, model_line,
                     where=model_line < overall_mean,
                     alpha=0.15, color=_COLORS["green"],
                     edgecolor="none", label="Early tokens overpay")
    ax1.fill_between(pos, overall_mean, model_line,
                     where=model_line > overall_mean,
                     alpha=0.15, color=_COLORS["red"],
                     edgecolor="none", label="Late tokens underpay")

    # Crossover point
    if beta != 0:
        crossover = (overall_mean - alpha) / beta
        if pos[0] <= crossover <= pos[-1]:
            ax1.axvline(crossover, color=_COLORS["purple"], linewidth=0.8,
                        linestyle="-.", alpha=0.7)
            ax1.annotate(f"Crossover @ pos {crossover:.0f}",
                         xy=(crossover, overall_mean),
                         xytext=(15, 20), textcoords="offset points",
                         fontsize=_FONT_SIZE["annotation"],
                         color=_COLORS["purple"],
                         arrowprops=dict(arrowstyle="->",
                                         color=_COLORS["purple"], lw=0.8))

    # Auto-clip y
    p1_y, p99_y = np.percentile(eng, 1), np.percentile(eng, 99)
    y_range = p99_y - p1_y
    y_lo = max(0, p1_y - 0.5 * y_range)
    y_hi = p99_y + 0.5 * y_range
    if y_hi < eng.max() * 0.85:
        ax1.set_ylim(y_lo, y_hi)

    ax1.set_xlabel("Token Position")
    ax1.set_ylabel("Energy per Token (mJ)")
    ax1.set_title("Uniform vs Position-Aware Token Pricing")
    ax1.legend(loc="upper left", fontsize=_FONT_SIZE["legend"] - 0.5)

    # ---- Panel 2: Cost weight w(p) ----
    ax2 = axes[1]

    ax2.axhline(1.0, color=_COLORS["gray"], linewidth=1.0, linestyle=":",
                alpha=0.7, label="Uniform weight = 1.0")
    ax2.plot(pos, weights, color=_COLORS["orange"], linewidth=1.8,
             label="Position-aware weight w(p)")

    ax2.fill_between(pos, 1.0, weights, where=weights < 1.0,
                     alpha=0.15, color=_COLORS["green"], edgecolor="none")
    ax2.fill_between(pos, 1.0, weights, where=weights > 1.0,
                     alpha=0.15, color=_COLORS["red"], edgecolor="none")

    ax2.set_xlabel("Token Position")
    ax2.set_ylabel("Cost Weight w(p)")
    ax2.set_title("Position-Aware Cost Weight: w(p) = E_model(p) / E_mean")
    ax2.legend(loc="upper left", fontsize=_FONT_SIZE["legend"])

    # Annotate endpoints
    ax2.annotate(f"w={weights[0]:.2f}x",
                 xy=(pos[0], weights[0]),
                 xytext=(10, -15), textcoords="offset points",
                 fontsize=_FONT_SIZE["annotation"], color=_COLORS["green"])
    ax2.annotate(f"w={weights[-1]:.2f}x",
                 xy=(pos[-1], weights[-1]),
                 xytext=(-40, 10), textcoords="offset points",
                 fontsize=_FONT_SIZE["annotation"], color=_COLORS["red"])

    # ---- Panel 3: Quartile fairness bar chart ----
    ax3 = axes[2]

    x = np.arange(4)
    bar_w = 0.25
    bars1 = ax3.bar(x - bar_w, q_uniform_frac, bar_w,
                    color=_COLORS["gray"], alpha=0.7, edgecolor="white",
                    linewidth=0.5, label="Uniform pricing (25% each)")
    bars2 = ax3.bar(x, q_actual_frac, bar_w,
                    color=_COLORS["blue"], alpha=0.8, edgecolor="white",
                    linewidth=0.5, label="Actual energy share")
    bars3 = ax3.bar(x + bar_w, q_model_frac, bar_w,
                    color=_COLORS["orange"], alpha=0.8, edgecolor="white",
                    linewidth=0.5, label="Position-aware pricing")

    # Value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax3.annotate(f"{h:.1f}%",
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom",
                         fontsize=_FONT_SIZE["annotation"] - 0.5)

    ax3.set_xticks(x)
    ax3.set_xticklabels(q_labels)
    ax3.set_ylabel("Share of Total Cost (%)")
    ax3.set_title("Cost Fairness by Quartile: Who Pays What?")
    ax3.legend(loc="upper left", fontsize=_FONT_SIZE["legend"])
    ax3.set_ylim(0, max(max(q_actual_frac), max(q_model_frac)) * 1.25)

    # ---- Caption ----
    q4_q1_ratio = eng[-q_size:].mean() / eng[:q_size].mean()
    max_subsidy = (1 - weights[0]) * 100
    max_penalty = (weights[-1] - 1) * 100
    caption = (
        f"Linear energy model: E(p) = {alpha:.0f} + {beta:.4f}p mJ  |  "
        f"Q4/Q1 energy ratio = {q4_q1_ratio:.2f}x  |  "
        f"Early tokens overpay by {max_subsidy:.0f}%, "
        f"late tokens underpay by {max_penalty:.0f}% under uniform pricing"
    )
    _add_caption(fig, caption, y=-0.02)

    _savefig(fig, os.path.join(output_dir, "token_cost_model.png"))

    # ---- Print cost model summary ----
    print(f"\n{'='*60}")
    print(f"  POSITION-AWARE TOKEN COST MODEL")
    print(f"{'='*60}")
    print(f"  Linear model:  E(p) = {alpha:.2f} + {beta:.4f} * p  (mJ)")
    print(f"  R-squared:     {r2:.4f}")
    print(f"  Mean energy:   {overall_mean:.2f} mJ/token")
    print(f"  Q4/Q1 ratio:   {q4_q1_ratio:.3f}x")
    print(f"")
    print(f"  Cost weight range:")
    print(f"    Position {int(pos[0]):>6d}:  w = {weights[0]:.3f}x  "
          f"(pays {weights[0]*100:.1f}% of uniform)")
    print(f"    Position {int(pos[-1]):>6d}:  w = {weights[-1]:.3f}x  "
          f"(pays {weights[-1]*100:.1f}% of uniform)")
    if beta != 0:
        crossover = (overall_mean - alpha) / beta
        if pos[0] <= crossover <= pos[-1]:
            print(f"    Crossover:     position {crossover:.0f}")
    print(f"")
    print(f"  Quartile analysis (% of total cost):")
    print(f"    {'Quartile':<12s} {'Uniform':>10s} {'Actual':>10s} "
          f"{'Model':>10s} {'Unfairness':>12s}")
    for i in range(4):
        unfairness = q_actual_frac[i] - q_uniform_frac[i]
        sign = "+" if unfairness > 0 else ""
        print(f"    {q_labels[i].replace(chr(10),' '):<12s} "
              f"{q_uniform_frac[i]:>9.1f}% {q_actual_frac[i]:>9.1f}% "
              f"{q_model_frac[i]:>9.1f}% {sign}{unfairness:>10.1f}pp")
    print(f"")
    print(f"  Formula for fair pricing:")
    print(f"    cost(token_at_pos_p) = base_price * w(p)")
    print(f"    where w(p) = ({alpha:.1f} + {beta:.4f} * p) / {overall_mean:.1f}")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════
#  Main entry point
# ══════════════════════════════════════════════════════════════

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

    # Auto-detect mode
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
            print(f"⚠ No token energy data found in {data_dir}, skipping.")
            return False

    if mode == "stream":
        pos_data = read_csv(stream_pos_csv)
        sample_csv = os.path.join(data_dir, "token_energy_stream_per_sample.csv")
        sample_data = read_csv(sample_csv)

        if not pos_data:
            print(f"⚠ No data in {stream_pos_csv}, skipping.")
            return False

        print(f"\n📊 Generating STREAM visualizations...")
        print(f"   Data:    {data_dir}")
        print(f"   Output:  {output_dir}")
        print(f"   Points:  {len(pos_data)} positions, {len(sample_data)} samples\n")

        plot_batch_energy_curve(pos_data, output_dir)
        plot_batch_step_vs_token_energy(pos_data, output_dir)
        plot_cumulative_energy(pos_data, output_dir)
        plot_energy_distribution(pos_data, output_dir)
        plot_token_cost_model(pos_data, output_dir)
        plot_ttft_latency(sample_data, output_dir)
        print_batch_summary(pos_data, [])

    elif mode == "batch":
        pos_data_raw = read_csv(batch_pos_csv)
        step_raw_csv = os.path.join(data_dir, "token_energy_batch_step_raw.csv")
        step_raw_data = read_csv(step_raw_csv)
        sample_csv = os.path.join(data_dir, "token_energy_batch_per_sample.csv")
        sample_data = read_csv(sample_csv)

        if not pos_data_raw:
            print(f"⚠ No data in {batch_pos_csv}, skipping.")
            return False

        # Filter: decode only (skip prefill & prefill_tail)
        pos_data = [r for r in pos_data_raw
                    if r.get("phase", "decode") == "decode"]

        print(f"\n📊 Generating BATCH visualizations...")
        print(f"   Data:    {data_dir}")
        print(f"   Output:  {output_dir}")
        print(f"   Points:  {len(pos_data)} decode / {len(pos_data_raw)} total, "
              f"{len(step_raw_data)} steps, {len(sample_data)} samples\n")

        plot_batch_energy_curve(pos_data, output_dir)
        plot_batch_step_vs_token_energy(pos_data, output_dir)
        plot_cumulative_energy(pos_data, output_dir)
        plot_energy_distribution(pos_data, output_dir)
        plot_token_cost_model(pos_data, output_dir)
        print_batch_summary(pos_data, step_raw_data)

    else:
        pos_data = read_csv(seq_pos_csv)
        sample_csv = os.path.join(data_dir, "token_energy_per_sample.csv")
        raw_csv = os.path.join(data_dir, "token_energy_raw.csv")
        sample_data = read_csv(sample_csv)
        raw_data = read_csv(raw_csv)

        if not pos_data:
            print(f"⚠ No data in {seq_pos_csv}, skipping.")
            return False

        print(f"\n📊 Generating SEQUENTIAL visualizations...")
        print(f"   Data:    {data_dir}")
        print(f"   Output:  {output_dir}")
        print(f"   Points:  {len(pos_data)} positions, "
              f"{len(sample_data)} samples, {len(raw_data)} raw\n")

        plot_mean_energy_curve(pos_data, output_dir)
        plot_energy_heatmap(raw_data, output_dir)
        plot_cumulative_energy(pos_data, output_dir)
        plot_sample_comparison(sample_data, output_dir)
        plot_energy_distribution(pos_data, output_dir)
        plot_token_cost_model(pos_data, output_dir)
        print_summary_table(pos_data, sample_data)

    print(f"\n✅ All figures saved to {output_dir}/")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Visualize per-token energy benchmark results")
    _project_root = os.path.join(os.path.dirname(__file__), "..")
    parser.add_argument("--experiment_dir", type=str, default=None,
                        help="Path to an experiment output directory. "
                             "If provided, data_dir = experiment_dir, "
                             "output_dir = experiment_dir/figures.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing token_energy_*.csv files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save output figures")
    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "sequential", "batch", "stream"],
                        help="Visualization mode")
    args = parser.parse_args()

    if args.experiment_dir:
        data_dir = args.experiment_dir
        output_dir = os.path.join(args.experiment_dir, "figures")
    else:
        data_dir = args.data_dir or os.path.join(_project_root, "output")
        output_dir = args.output_dir or os.path.join(_project_root, "figures")

    success = run_visualization(data_dir=data_dir, output_dir=output_dir,
                                mode=args.mode)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
