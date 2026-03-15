#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulate_scheduling.py — Position-Aware Scheduling Policy Simulator
                         with Real-World Economic Analysis

Compares scheduling strategies for LLM serving:

  1. FCFS (First-Come-First-Served):
     Default vLLM behaviour. All active requests decode every step.
     New requests wait when batch is full.

  2. PAP (Position-Aware Preemptive):
     When batch is full and a new request arrives, the request with
     the *highest* position (most expensive, least UX impact) is
     paused for a short burst to make room.

  3. PB (Position-Budget — Pure Swap):
     **New**: Each step has an energy cost cap. If the total step cost
     exceeds the cap, high-position requests are swapped 1:1 with
     new requests from the waiting queue (position 0). In continuous
     serving (stream mode), the batch stays FULL — no throughput loss.
     PB only changes WHICH tokens are generated, not HOW MANY.

The energy model  E(p) = α + β·p  is calibrated from real experiments.

Real-world economics are computed using configurable pricing tiers:
  - GPU amortization ($/hour)
  - Electricity cost ($/kWh)
  - Revenue per output token ($/M tokens)

Usage:
    python scripts/simulate_scheduling.py
    python scripts/simulate_scheduling.py --data_csv <per_position.csv>
    python scripts/simulate_scheduling.py --output_dir output/scheduling_sim
    python scripts/simulate_scheduling.py --arrival_rate 0.13 --max_batch 60
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
except ImportError:
    print("ERROR: matplotlib required. pip install matplotlib")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════
#  Style (NeurIPS/ICLR style)
# ═══════════════════════════════════════════════════════════

_COLORS = {
    "blue":    "#4878D0",
    "orange":  "#EE854A",
    "green":   "#6ACC64",
    "red":     "#D65F5F",
    "purple":  "#956CB4",
    "gray":    "#8C8C8C",
    "teal":    "#17BECF",
    "light_blue": "#A6CEE3",
    "light_orange": "#FDBF6F",
}
_FIG_DPI = 300
_FS = {"title": 13, "label": 11, "tick": 9.5, "legend": 9,
       "annotation": 8.5, "caption": 8.5}


def _apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "font.size": _FS["tick"],
        "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
        "axes.labelsize": _FS["label"], "axes.titlesize": _FS["title"],
        "axes.titleweight": "bold", "axes.titlepad": 10,
        "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": "#E0E0E0", "grid.linewidth": 0.5,
        "grid.linestyle": "--", "grid.alpha": 0.7,
        "xtick.labelsize": _FS["tick"], "ytick.labelsize": _FS["tick"],
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "legend.fontsize": _FS["legend"], "legend.frameon": True,
        "legend.framealpha": 0.9, "legend.edgecolor": "#CCCCCC",
        "figure.dpi": _FIG_DPI, "savefig.dpi": _FIG_DPI,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        "lines.linewidth": 1.5, "lines.markersize": 4,
    })


_apply_style()


def _add_caption(fig, text, y=-0.03):
    fig.text(0.5, y, text, ha="center", va="top",
             fontsize=_FS["caption"], fontstyle="italic",
             color="#555555", wrap=True)


def _savefig(fig, path):
    fig.savefig(path, dpi=_FIG_DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  saved: {path}")


# ═══════════════════════════════════════════════════════════
#  Energy Model
# ═══════════════════════════════════════════════════════════

@dataclass
class EnergyModel:
    """Linear energy model: E(p) = alpha + beta * p  (mJ per token)."""
    alpha: float = 732.8     # base energy at position 0 (mJ)
    beta: float = 0.1005     # energy increment per position (mJ)
    prefill_energy: float = 1851.0  # energy for position 0 (prefill tail)

    def __call__(self, position: int) -> float:
        if position <= 0:
            return self.prefill_energy
        return self.alpha + self.beta * position

    def step_energy(self, positions: list) -> float:
        """Total energy for one decode step with given request positions."""
        return sum(self(p) for p in positions)

    @property
    def mean_energy_at_midpoint(self) -> float:
        """E(250) as a typical mid-sequence cost."""
        return self(250)

    @classmethod
    def from_csv(cls, csv_path: str) -> "EnergyModel":
        """Fit energy model from experimental per-position CSV data."""
        with open(csv_path) as f:
            data = [r for r in csv.DictReader(f)
                    if r.get("phase", "decode") == "decode"]
        positions = np.array([int(r["position"]) for r in data])
        means = np.array([float(r["mean_energy_mj"]) for r in data])
        mask = positions >= 2
        if mask.sum() < 10:
            return cls()
        pos, eng = positions[mask].astype(float), means[mask]
        coeffs = np.polyfit(pos, eng, 1)
        prefill_e = means[0] if len(means) > 0 else 1851.0
        return cls(alpha=coeffs[1], beta=coeffs[0],
                   prefill_energy=prefill_e)


# ═══════════════════════════════════════════════════════════
#  Economic Model — Real-World AI Serving Cost
# ═══════════════════════════════════════════════════════════

@dataclass
class EconomicModel:
    """
    Real-world AI serving economics.

    Cost structure:
      - GPU amortization: dominant (60-70% of total)
      - Electricity: small (3-8% of total)
      - Networking/cooling/ops: remainder

    Revenue:
      - Per output token, uniform pricing (industry standard)
    """
    # Infrastructure (V100-class GPU)
    gpu_cost_per_hour: float = 0.80        # USD (amortized purchase+maintenance)
    electricity_per_kwh: float = 0.10      # USD
    gpu_idle_power_w: float = 80.0         # watts (GPU idle)
    cooling_overhead: float = 1.3          # PUE (Power Usage Effectiveness)
    ops_overhead_per_hour: float = 0.10    # USD (network, storage, staff)
    step_duration_s: float = 0.030         # seconds per decode step (30ms)

    # Revenue tiers ($/M output tokens)
    pricing_tiers: Dict[str, float] = field(default_factory=lambda: {
        "premium":  10.00,   # GPT-4o / Claude Opus class
        "standard":  1.00,   # GPT-4o-mini / Claude Haiku class
        "budget":    0.27,   # DeepSeek-V3 class
        "economy":   0.14,   # Open-source hosting / batch API
    })

    def cost_per_step(self, step_energy_mj: float) -> dict:
        """
        Compute operator cost for one decode step.

        Returns dict with cost breakdown in USD.
        """
        dt = self.step_duration_s
        # GPU amortization (fixed cost per unit time)
        gpu_cost = self.gpu_cost_per_hour / 3600 * dt
        # Electricity from measured GPU energy
        gpu_energy_j = step_energy_mj / 1000  # mJ → J
        # Add idle power (always consumed)
        idle_energy_j = self.gpu_idle_power_w * dt
        total_energy_j = gpu_energy_j + idle_energy_j
        # Apply PUE for cooling
        total_energy_kwh = total_energy_j / 3_600_000 * self.cooling_overhead
        electricity_cost = total_energy_kwh * self.electricity_per_kwh
        # Ops overhead
        ops_cost = self.ops_overhead_per_hour / 3600 * dt
        total = gpu_cost + electricity_cost + ops_cost
        return {
            "gpu_amort": gpu_cost,
            "electricity": electricity_cost,
            "ops": ops_cost,
            "total": total,
            "energy_j": total_energy_j,
        }

    def revenue_per_token(self, tier: str = "standard") -> float:
        """Revenue per token in USD."""
        return self.pricing_tiers.get(tier, 1.0) / 1_000_000

    def hourly_report(self, total_tokens: int, total_energy_mj: float,
                      total_steps: int) -> dict:
        """
        Compute hourly economics extrapolation.

        Given simulation results, extrapolate to per-hour figures.
        """
        sim_duration_s = total_steps * self.step_duration_s
        sim_duration_h = sim_duration_s / 3600
        scale = 1.0 / sim_duration_h if sim_duration_h > 0 else 0

        tokens_per_hour = total_tokens * scale
        steps_per_hour = total_steps * scale

        # Cost
        energy_per_step_avg = total_energy_mj / max(1, total_steps)
        cost_per_step = self.cost_per_step(energy_per_step_avg)
        total_cost_hour = cost_per_step["total"] * steps_per_hour
        electricity_hour = cost_per_step["electricity"] * steps_per_hour
        gpu_cost_hour = cost_per_step["gpu_amort"] * steps_per_hour

        report = {
            "tokens_per_hour": tokens_per_hour,
            "throughput_tok_per_s": total_tokens / max(1, sim_duration_s),
            "total_cost_per_hour": total_cost_hour,
            "gpu_amort_per_hour": gpu_cost_hour,
            "electricity_per_hour": electricity_hour,
            "cost_per_M_tokens": total_cost_hour / max(1, tokens_per_hour) * 1e6,
        }

        # Revenue and profit per tier
        for tier, price in self.pricing_tiers.items():
            rev = tokens_per_hour * price / 1e6
            profit = rev - total_cost_hour
            margin = profit / rev * 100 if rev > 0 else -100
            report[f"revenue_{tier}"] = rev
            report[f"profit_{tier}"] = profit
            report[f"margin_{tier}"] = margin

        return report


# ═══════════════════════════════════════════════════════════
#  Request & Simulation State
# ═══════════════════════════════════════════════════════════

@dataclass
class Request:
    """A single inference request."""
    id: int
    arrival_step: int
    target_length: int
    current_position: int = 0
    first_token_step: int = -1
    completion_step: int = -1
    paused_steps: int = 0
    consecutive_idle: int = 0
    resume_at_step: int = -1
    is_active: bool = False
    is_paused: bool = False
    is_done: bool = False
    is_abandoned: bool = False
    energy_consumed_mj: float = 0.0

    @property
    def ttft(self) -> int:
        if self.first_token_step < 0:
            return -1
        return self.first_token_step - self.arrival_step

    @property
    def total_time(self) -> int:
        if self.completion_step < 0:
            return -1
        return self.completion_step - self.arrival_step

    @property
    def generation_delay(self) -> float:
        if self.completion_step < 0:
            return -1
        return self.total_time - self.target_length


@dataclass
class SimResult:
    """Results from one simulation run."""
    policy_name: str
    total_steps: int = 0
    total_energy_mj: float = 0.0
    energy_per_step: list = field(default_factory=list)
    batch_size_per_step: list = field(default_factory=list)
    avg_position_per_step: list = field(default_factory=list)
    position_sum_per_step: list = field(default_factory=list)
    completed_requests: list = field(default_factory=list)
    abandoned_requests: list = field(default_factory=list)
    step_positions: list = field(default_factory=list)
    wasted_energy_mj: float = 0.0
    total_tokens_generated: int = 0  # including partial (abandoned)


# ═══════════════════════════════════════════════════════════
#  Scheduler Policies
# ═══════════════════════════════════════════════════════════

def schedule_fcfs(active: list, waiting: list, paused: list,
                  max_batch: int, **kwargs) -> tuple:
    """
    FCFS: First-Come-First-Served.
    Active requests keep going. New requests fill remaining slots.
    No preemption.
    """
    while len(active) < max_batch and waiting:
        req = waiting.pop(0)
        req.is_active = True
        active.append(req)
    return active, waiting, paused


def schedule_pap(active: list, waiting: list, paused: list,
                 max_batch: int, current_step: int = 0, **kwargs) -> tuple:
    """
    PAP: Position-Aware Preemptive (Short-Burst).
    Only preempts when batch is full AND a new request is waiting.
    Maintains throughput but improves TTFT for new arrivals.
    """
    burst_len = 10
    max_over = 3

    # Force-resume expired bursts
    expired = sorted(
        [r for r in paused if r.resume_at_step >= 0
         and current_step >= r.resume_at_step],
        key=lambda r: -r.paused_steps
    )
    for req in expired:
        if len(active) < max_batch + max_over:
            paused.remove(req)
            req.is_paused = False
            req.resume_at_step = -1
            req.is_active = True
            active.append(req)
        else:
            req.resume_at_step = current_step + 1

    # Fill empty slots
    while len(active) < max_batch:
        if paused:
            paused.sort(key=lambda r: -r.paused_steps)
            req = paused.pop(0)
            req.is_paused = False
            req.resume_at_step = -1
            req.is_active = True
            active.append(req)
        elif waiting:
            req = waiting.pop(0)
            req.is_active = True
            active.append(req)
        else:
            break

    # Short-burst preemption
    has_pending = any(r.resume_at_step >= 0 for r in paused)
    if waiting and len(active) == max_batch and not has_pending:
        candidates = sorted(active, key=lambda r: -r.current_position)
        if candidates:
            highest = candidates[0]
            avg_pos = np.mean([r.current_position for r in active])
            remaining_frac = 1 - highest.current_position / max(1, highest.target_length)
            if (highest.current_position > avg_pos * 1.5
                    and highest.current_position > 100
                    and remaining_frac > 0.10):
                active.remove(highest)
                highest.is_active = False
                highest.is_paused = True
                highest.resume_at_step = current_step + burst_len
                paused.append(highest)
                new_req = waiting.pop(0)
                new_req.is_active = True
                active.append(new_req)

    return active, waiting, paused


# ─── PB state (persists across steps via module-level dict) ───
_pb_state = {
    "ema_sum": None,       # EMA of position sum
    "ema_ratio": None,     # EMA of max/min ratio
    "ema_alpha": 0.05,     # EMA smoothing factor
}

def _reset_pb_state():
    _pb_state["ema_sum"] = None
    _pb_state["ema_ratio"] = None

def schedule_position_budget(active: list, waiting: list, paused: list,
                             max_batch: int, energy_model=None,
                             step_cost_cap: float = None,
                             current_step: int = 0, **kwargs) -> tuple:
    """
    PB: Position-Budget — Dual-Gate Swap with Dynamic Thresholds.

    Uses TWO complementary gates (both must trigger) to decide swaps:

      Gate 1 — Position Sum:
        Total position sum of active batch > dynamic threshold.
        Threshold = EMA(position_sum) × sum_cap_factor.
        sum_cap_factor shrinks when queue is deeper → more aggressive.

      Gate 2 — Spread Ratio:
        max_position / median_position of active batch > dynamic threshold.
        Threshold = base_ratio / (1 + queue_pressure × sensitivity).
        More waiting requests → lower threshold → swap sooner.

    Both thresholds are dynamically updated via Exponential Moving Average
    (EMA) to adapt to the workload's natural position distribution, avoiding
    fixed magic numbers that break under different loads.

    Additional safeguards:
      - Cooldown protection: recently-resumed requests immune to swap.
      - Anti-starvation: force-resume after max_pause_steps.
      - Completion protection: requests >85% done can't be swapped.
      - Max 2 swaps per step to limit disruption.

    Net effect in overload:
      - Batch size: unchanged (always max_batch) → throughput preserved
      - TTFT: improved (new requests get served sooner)
      - Step energy: reduced (lower avg position in batch)
      - Completion time: slightly longer for swapped-out requests
    """
    max_pause_steps = 80     # force-resume after this many paused steps
    swap_cooldown = 25       # can't be swapped again for N steps after resume
    max_swaps_per_step = 2   # allow up to 2 swaps per step

    swaps_this_step = 0
    alpha = _pb_state["ema_alpha"]

    # ─── Step 0: Compute batch statistics for dual-gate ───
    positions = [r.current_position for r in active] if active else []
    if positions:
        pos_sum = sum(positions)
        pos_max = max(positions)
        pos_min = min(p for p in positions if p > 0) if any(p > 0 for p in positions) else 1
        pos_median = float(np.median(positions))
        spread_ratio = pos_max / max(1, pos_median) if pos_median > 0 else 1.0

        # Update EMA of position sum
        if _pb_state["ema_sum"] is None:
            _pb_state["ema_sum"] = pos_sum
        else:
            _pb_state["ema_sum"] = alpha * pos_sum + (1 - alpha) * _pb_state["ema_sum"]

        # Update EMA of spread ratio
        if _pb_state["ema_ratio"] is None:
            _pb_state["ema_ratio"] = spread_ratio
        else:
            _pb_state["ema_ratio"] = alpha * spread_ratio + (1 - alpha) * _pb_state["ema_ratio"]
    else:
        pos_sum = 0
        spread_ratio = 1.0

    # ─── Dynamic threshold computation ───
    queue_pressure = len(waiting) / max(1, max_batch)  # 0.0 ~ N
    queue_pressure = min(queue_pressure, 3.0)  # cap at 3x

    # Gate 1: Position sum threshold
    # When queue is empty: sum_cap = EMA × 1.0 (lenient, rarely trigger)
    # When queue is 1× batch: sum_cap = EMA × 0.7 (moderate)
    # When queue is 2× batch: sum_cap = EMA × 0.5 (aggressive)
    sum_cap_factor = max(0.4, 1.0 - 0.2 * queue_pressure)
    sum_threshold = (_pb_state["ema_sum"] or pos_sum) * sum_cap_factor

    # Gate 2: Spread ratio threshold
    # When queue is empty: ratio_threshold = 5.0 (very lenient)
    # When queue is 1× batch: ratio_threshold = 3.0
    # When queue is 2× batch: ratio_threshold = 2.0
    base_ratio = 5.0
    ratio_sensitivity = 1.5
    ratio_threshold = max(1.5, base_ratio / (1 + ratio_sensitivity * queue_pressure))

    # Dual gate: BOTH must be true to enable swapping
    gate1_triggered = pos_sum > sum_threshold
    gate2_triggered = spread_ratio > ratio_threshold
    swap_enabled = gate1_triggered and gate2_triggered and len(waiting) > 0

    # ─── Step 1: Force-resume starved requests with cooldown ───
    starved = [r for r in paused if r.paused_steps >= max_pause_steps]
    starved.sort(key=lambda r: -r.paused_steps)
    for req in starved:
        if len(active) < max_batch:
            paused.remove(req)
            req.is_paused = False
            req.is_active = True
            req.resume_at_step = current_step + swap_cooldown
            active.append(req)
        else:
            # Swap with highest-position active (no cooldown)
            active.sort(key=lambda r: -r.current_position)
            swap_target = None
            for r in active:
                if (r.current_position > req.current_position
                        and getattr(r, 'resume_at_step', 0) <= current_step):
                    swap_target = r
                    break
            if swap_target:
                active.remove(swap_target)
                swap_target.is_active = False
                swap_target.is_paused = True
                swap_target.paused_steps = 0
                paused.append(swap_target)
                paused.remove(req)
                req.is_paused = False
                req.is_active = True
                req.resume_at_step = current_step + swap_cooldown
                active.append(req)
            else:
                # Can't swap for starved req — leave in paused for now,
                # it will be picked up when a slot opens
                pass

    # ─── Step 2: Fill empty slots from PAUSED (cheapest first) ───
    while len(active) < max_batch and paused:
        paused.sort(key=lambda r: r.current_position)
        req = paused.pop(0)
        req.is_paused = False
        req.is_active = True
        req.resume_at_step = current_step + swap_cooldown
        active.append(req)

    # ─── Step 3: Fill remaining empty slots from WAITING ───
    while len(active) < max_batch and waiting:
        req = waiting.pop(0)
        req.is_active = True
        active.append(req)

    # ─── Step 4: Dual-gate swap ───
    # Only swap when BOTH gates trigger (sum too high AND spread too wide)
    if swap_enabled and len(active) >= max_batch:
        avg_pos = np.mean(positions) if positions else 0

        active.sort(key=lambda r: -r.current_position)
        while (waiting and swaps_this_step < max_swaps_per_step):
            candidate = None
            for r in active:
                frac = r.current_position / max(1, r.target_length)
                cooldown_ok = getattr(r, 'resume_at_step', 0) <= current_step
                # Must be above average AND not near completion AND no cooldown
                if (r.current_position > avg_pos * 1.3
                        and frac < 0.85
                        and cooldown_ok):
                    candidate = r
                    break
            if candidate is None:
                break

            # Execute swap
            active.remove(candidate)
            candidate.is_active = False
            candidate.is_paused = True
            paused.append(candidate)

            new_req = waiting.pop(0)
            new_req.is_active = True
            active.append(new_req)

            swaps_this_step += 1
            active.sort(key=lambda r: -r.current_position)

    return active, waiting, paused


POLICIES = {
    "FCFS": schedule_fcfs,
    "PAP": schedule_pap,
    "PB": schedule_position_budget,
}


# ═══════════════════════════════════════════════════════════
#  Simulation Engine
# ═══════════════════════════════════════════════════════════

def run_simulation(
    policy_name: str,
    energy_model: EnergyModel,
    arrival_rate: float = 0.12,
    max_batch: int = 60,
    max_steps: int = 5000,
    seq_length_mean: int = 500,
    seq_length_std: int = 300,
    seq_length_min: int = 50,
    seq_length_max: int = 12000,
    cost_cap_factor: float = 0.85,
    user_patience: int = 300,
    seed: int = 42,
) -> SimResult:
    """Run a discrete-event simulation of LLM serving."""
    rng = np.random.RandomState(seed)
    scheduler = POLICIES[policy_name]

    # Compute the cost cap for PB policy
    # cap = factor × max_batch × E(mean_position)
    mid_pos = seq_length_mean // 2
    e_mid = energy_model(mid_pos)
    step_cost_cap = cost_cap_factor * max_batch * e_mid

    # Reset PB state for this simulation run
    _reset_pb_state()

    result = SimResult(policy_name=policy_name)

    active = []
    waiting = []
    paused = []
    all_requests = []
    next_id = 0

    for step in range(max_steps):
        # 1. Generate new arrivals (Poisson)
        n_arrivals = rng.poisson(arrival_rate)
        for _ in range(n_arrivals):
            length = int(np.clip(
                rng.normal(seq_length_mean, seq_length_std),
                seq_length_min, seq_length_max))
            req = Request(id=next_id, arrival_step=step,
                          target_length=length)
            waiting.append(req)
            all_requests.append(req)
            next_id += 1

        # 2. Run scheduler
        active, waiting, paused = scheduler(
            active, waiting, paused, max_batch,
            energy_model=energy_model,
            step_cost_cap=step_cost_cap,
            current_step=step)

        # 3. Execute one decode step for all active requests
        step_energy = 0.0
        positions_snapshot = []
        completed_this_step = []

        for req in active:
            if req.first_token_step < 0:
                req.first_token_step = step
            token_e = energy_model(req.current_position)
            step_energy += token_e
            req.energy_consumed_mj += token_e
            req.consecutive_idle = 0
            positions_snapshot.append(req.current_position)
            req.current_position += 1
            result.total_tokens_generated += 1
            if req.current_position >= req.target_length:
                req.is_done = True
                req.is_active = False
                req.completion_step = step
                completed_this_step.append(req)

        # Track idle / paused steps
        for req in paused:
            req.paused_steps += 1
            req.consecutive_idle += 1
        for req in waiting:
            req.consecutive_idle += 1

        # Remove completed from active
        for req in completed_this_step:
            if req in active:
                active.remove(req)
            result.completed_requests.append(req)

        # 4. Check for user abandonment
        abandoned_this_step = []
        for queue in [waiting, paused]:
            for req in queue[:]:
                if req.consecutive_idle >= user_patience:
                    req.is_abandoned = True
                    req.is_paused = False
                    req.completion_step = step
                    queue.remove(req)
                    abandoned_this_step.append(req)

        for req in abandoned_this_step:
            result.abandoned_requests.append(req)
            result.wasted_energy_mj += req.energy_consumed_mj

        # Record step metrics
        result.energy_per_step.append(step_energy)
        result.batch_size_per_step.append(len(positions_snapshot))
        pos_sum = sum(positions_snapshot) if positions_snapshot else 0
        result.position_sum_per_step.append(pos_sum)
        result.avg_position_per_step.append(
            np.mean(positions_snapshot) if positions_snapshot else 0)
        result.step_positions.append(positions_snapshot[:])
        result.total_energy_mj += step_energy

    result.total_steps = max_steps
    return result


# ═══════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════

def compute_metrics(result: SimResult, econ: EconomicModel = None) -> dict:
    """Compute summary metrics from simulation results."""
    completed = result.completed_requests
    abandoned = result.abandoned_requests

    energy_arr = np.array(result.energy_per_step) if result.energy_per_step else np.array([0])
    batch_arr = np.array(result.batch_size_per_step) if result.batch_size_per_step else np.array([0])

    # Tokens from completed requests only (= revenue-generating tokens)
    useful_tokens = sum(r.current_position for r in completed)
    # All tokens including abandoned
    all_tokens = result.total_tokens_generated

    base = {
        "policy": result.policy_name,
        "completed": len(completed),
        "abandoned": len(abandoned),
        "total_tokens": all_tokens,
        "useful_tokens": useful_tokens,
        "total_energy_J": result.total_energy_mj / 1000,
        "wasted_energy_J": result.wasted_energy_mj / 1000,
        "useful_energy_J": (result.total_energy_mj - result.wasted_energy_mj) / 1000,
        "avg_step_energy_mJ": float(np.mean(energy_arr)),
        "peak_step_energy_mJ": float(np.max(energy_arr)),
        "p99_step_energy_mJ": float(np.percentile(energy_arr, 99)) if len(energy_arr) > 10 else float(np.max(energy_arr)),
        "energy_std_per_step_mJ": float(np.std(energy_arr)),
        "avg_batch_size": float(np.mean(batch_arr)),
        "batch_utilization": float(np.mean(batch_arr)) / max(1, max(batch_arr)) * 100,
        "avg_position_in_batch": float(np.mean(result.avg_position_per_step))
            if result.avg_position_per_step else 0,
        "avg_position_sum": float(np.mean(result.position_sum_per_step))
            if result.position_sum_per_step else 0,
    }

    # Energy efficiency
    base["energy_per_useful_token_mJ"] = result.total_energy_mj / max(1, useful_tokens)
    base["energy_efficiency"] = useful_tokens / (result.total_energy_mj / 1000) if result.total_energy_mj > 0 else 0

    if not completed:
        base.update({
            "avg_energy_per_token_mJ": 0,
            "avg_ttft_steps": 0, "median_ttft_steps": 0,
            "p95_ttft_steps": 0, "p99_ttft_steps": 0,
            "avg_total_time_steps": 0, "avg_paused_steps": 0,
            "max_paused_steps": 0, "avg_delay_steps": 0,
        })
    else:
        ttfts = [r.ttft for r in completed if r.ttft >= 0]
        total_times = [r.total_time for r in completed if r.total_time >= 0]
        paused_steps = [r.paused_steps for r in completed]
        delays = [r.generation_delay for r in completed if r.generation_delay >= 0]

        base.update({
            "avg_energy_per_token_mJ": result.total_energy_mj / max(1, all_tokens),
            "avg_ttft_steps": float(np.mean(ttfts)) if ttfts else 0,
            "median_ttft_steps": float(np.median(ttfts)) if ttfts else 0,
            "p95_ttft_steps": float(np.percentile(ttfts, 95)) if len(ttfts) >= 20 else (max(ttfts) if ttfts else 0),
            "p99_ttft_steps": float(np.percentile(ttfts, 99)) if len(ttfts) >= 100 else (max(ttfts) if ttfts else 0),
            "avg_total_time_steps": float(np.mean(total_times)) if total_times else 0,
            "avg_paused_steps": float(np.mean(paused_steps)),
            "max_paused_steps": max(paused_steps),
            "avg_delay_steps": float(np.mean(delays)) if delays else 0,
        })

    # Economic metrics
    if econ:
        report = econ.hourly_report(
            total_tokens=useful_tokens,
            total_energy_mj=result.total_energy_mj,
            total_steps=result.total_steps)
        base["econ"] = report

    return base


# ═══════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════

def plot_comparison(results: dict, output_dir: str,
                    energy_model: EnergyModel, econ: EconomicModel):
    """Generate comparison plots for all scheduling policies."""
    os.makedirs(output_dir, exist_ok=True)
    policies = list(results.keys())
    colors_map = {
        "FCFS": _COLORS["blue"],
        "PAP": _COLORS["orange"],
        "PB": _COLORS["green"],
    }

    # ════════ Figure 1: Step Energy + Position Sum + Batch Size ════════
    fig, axes = plt.subplots(3, 1, figsize=(7, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.2, 1],
                                          "hspace": 0.12})

    ax1 = axes[0]
    for name in policies:
        r = results[name]
        e = np.array(r.energy_per_step)
        window = max(5, len(e) // 100)
        if len(e) > window * 2:
            kernel = np.ones(window) / window
            smooth = np.convolve(e, kernel, mode="valid")
            x = np.arange(window // 2, window // 2 + len(smooth))
            ax1.plot(x, smooth / 1000, color=colors_map.get(name, _COLORS["gray"]),
                     linewidth=1.5, label=f"{name} (mov.avg w={window})")
    ax1.set_ylabel("Step Energy (J)")
    ax1.set_title("Per-Step Energy: FCFS vs Position-Aware Scheduling")
    ax1.legend(loc="upper right", fontsize=_FS["legend"])

    # Position sum (the metric user wants to control)
    ax2 = axes[1]
    for name in policies:
        r = results[name]
        psum = np.array(r.position_sum_per_step)
        window = max(5, len(psum) // 100)
        if len(psum) > window * 2:
            kernel = np.ones(window) / window
            smooth = np.convolve(psum, kernel, mode="valid")
            x = np.arange(window // 2, window // 2 + len(smooth))
            ax2.plot(x, smooth / 1000, color=colors_map.get(name, _COLORS["gray"]),
                     linewidth=1.5, label=name)
    ax2.set_ylabel("Position Sum\n(×1000)")
    ax2.legend(loc="upper right", fontsize=_FS["legend"])

    # Batch size
    ax3 = axes[2]
    for name in policies:
        r = results[name]
        bs = np.array(r.batch_size_per_step)
        window = max(5, len(bs) // 100)
        if len(bs) > window * 2:
            kernel = np.ones(window) / window
            smooth = np.convolve(bs, kernel, mode="valid")
            x = np.arange(window // 2, window // 2 + len(smooth))
            ax3.plot(x, smooth, color=colors_map.get(name, _COLORS["gray"]),
                     linewidth=1.5, label=name)
    ax3.set_ylabel("Batch Size")
    ax3.set_xlabel("Simulation Step")
    ax3.legend(loc="upper right", fontsize=_FS["legend"])

    _add_caption(fig,
                 "PB actively reduces batch size to cap step energy; "
                 "PAP only preempts when new requests arrive; FCFS never preempts.",
                 y=-0.02)
    _savefig(fig, os.path.join(output_dir, "scheduling_step_energy.png"))

    # ════════ Figure 2: Key Metrics Comparison ════════
    metrics_list = [compute_metrics(results[n], econ) for n in policies]

    fig, axes = plt.subplots(2, 3, figsize=(10, 6),
                             gridspec_kw={"hspace": 0.50, "wspace": 0.40})

    metric_specs = [
        ("peak_step_energy_mJ",    "Peak Step Energy (mJ)",    axes[0, 0]),
        ("avg_step_energy_mJ",     "Avg Step Energy (mJ)",     axes[0, 1]),
        ("energy_per_useful_token_mJ", "Energy/Useful Token (mJ)", axes[0, 2]),
        ("avg_ttft_steps",         "Avg TTFT (steps)",         axes[1, 0]),
        ("abandoned",              "Abandoned Requests",       axes[1, 1]),
        ("useful_tokens",          "Useful Tokens (completed)", axes[1, 2]),
    ]

    x = np.arange(len(policies))
    for metric_key, ylabel, ax in metric_specs:
        vals = [m[metric_key] for m in metrics_list]
        bars = ax.bar(x, vals, color=[colors_map.get(n, _COLORS["gray"])
                                      for n in policies],
                      alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(policies, fontsize=_FS["tick"])
        ax.set_ylabel(ylabel, fontsize=_FS["label"] - 1)

        for bar, val in zip(bars, vals):
            label = f"{val:.0f}" if val > 100 else f"{val:.1f}"
            ax.annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=_FS["annotation"])

    fig.suptitle("Scheduling Policy Comparison", fontsize=_FS["title"] + 1,
                 fontweight="bold", y=1.01)

    # Delta caption
    if len(metrics_list) >= 2:
        fcfs_m = metrics_list[0]
        lines = []
        for m in metrics_list[1:]:
            if fcfs_m["avg_step_energy_mJ"] > 0:
                e_save = (1 - m["avg_step_energy_mJ"] / fcfs_m["avg_step_energy_mJ"]) * 100
            else:
                e_save = 0
            if fcfs_m["peak_step_energy_mJ"] > 0:
                peak_save = (1 - m["peak_step_energy_mJ"] / fcfs_m["peak_step_energy_mJ"]) * 100
            else:
                peak_save = 0
            lines.append(
                f"{m['policy']} vs FCFS: "
                f"avg energy {e_save:+.1f}%, peak {peak_save:+.1f}%"
            )
        _add_caption(fig, " | ".join(lines), y=-0.02)

    _savefig(fig, os.path.join(output_dir, "scheduling_comparison.png"))

    # ════════ Figure 3: TTFT Distribution ════════
    fig, axes = plt.subplots(1, len(policies), figsize=(4 * len(policies), 3.5),
                             sharey=True,
                             gridspec_kw={"wspace": 0.08})
    if len(policies) == 1:
        axes = [axes]

    for ax, name in zip(axes, policies):
        r = results[name]
        ttfts = [req.ttft for req in r.completed_requests if req.ttft >= 0]
        if not ttfts:
            ax.set_title(f"{name}\n(no completions)")
            continue
        p99 = np.percentile(ttfts, 99)
        clipped = [t for t in ttfts if t <= p99 * 1.5]
        n_bins = min(80, max(20, len(clipped) // 50))
        ax.hist(clipped, bins=n_bins,
                color=colors_map.get(name, _COLORS["gray"]),
                alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.axvline(np.mean(ttfts), color=_COLORS["red"], linewidth=1.2,
                   linestyle="--", label=f"Mean={np.mean(ttfts):.1f}")
        ax.axvline(np.median(ttfts), color=_COLORS["orange"], linewidth=1.2,
                   linestyle=":", label=f"Median={np.median(ttfts):.0f}")
        ax.set_title(name, fontsize=_FS["title"] - 1)
        ax.set_xlabel("TTFT (steps)")
        ax.legend(fontsize=_FS["legend"] - 0.5)
    axes[0].set_ylabel("Count")
    fig.suptitle("Time-to-First-Token Distribution", fontsize=_FS["title"],
                 fontweight="bold", y=1.02)
    _savefig(fig, os.path.join(output_dir, "scheduling_ttft_dist.png"))

    # ════════ Figure 4: Economic Analysis ════════
    _plot_economics(policies, metrics_list, output_dir, colors_map, econ)

    # ════════ Figure 5: Summary Table + Concept ════════
    _plot_summary_table(policies, metrics_list, output_dir, colors_map,
                        energy_model, econ)


def _plot_economics(policies, metrics_list, output_dir, colors_map, econ):
    """Generate the real-world economic comparison figure."""
    if not econ or not any("econ" in m for m in metrics_list):
        return

    tiers = list(econ.pricing_tiers.keys())

    fig, axes = plt.subplots(2, 2, figsize=(10, 7),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.35})

    x = np.arange(len(policies))
    bar_width = 0.7 / max(1, len(policies))

    # (0,0) Cost breakdown per hour
    ax = axes[0, 0]
    for i, name in enumerate(policies):
        m = metrics_list[i]
        ec = m.get("econ", {})
        gpu = ec.get("gpu_amort_per_hour", 0)
        elec = ec.get("electricity_per_hour", 0)
        ax.bar(i, gpu, bar_width * 2, color=_COLORS["gray"], alpha=0.7,
               label="GPU amortization" if i == 0 else "")
        ax.bar(i, elec, bar_width * 2, bottom=gpu,
               color=_COLORS["orange"], alpha=0.7,
               label="Electricity" if i == 0 else "")
        ax.annotate(f"${gpu + elec:.3f}",
                    xy=(i, gpu + elec), xytext=(0, 3),
                    textcoords="offset points", ha="center",
                    fontsize=_FS["annotation"])
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.set_ylabel("Cost ($/hour)")
    ax.set_title("Operator Cost Breakdown")
    ax.legend(fontsize=_FS["legend"] - 0.5)

    # (0,1) Throughput comparison
    ax = axes[0, 1]
    throughputs = [m.get("econ", {}).get("throughput_tok_per_s", 0)
                   for m in metrics_list]
    bars = ax.bar(x, throughputs, color=[colors_map.get(n, _COLORS["gray"])
                                         for n in policies],
                  alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, throughputs):
        ax.annotate(f"{val:.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=_FS["annotation"])
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("Effective Throughput\n(completed tokens only)")

    # (1,0) Cost per million useful tokens
    ax = axes[1, 0]
    cost_per_m = [m.get("econ", {}).get("cost_per_M_tokens", 0)
                  for m in metrics_list]
    bars = ax.bar(x, cost_per_m, color=[colors_map.get(n, _COLORS["gray"])
                                         for n in policies],
                  alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, cost_per_m):
        ax.annotate(f"${val:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=_FS["annotation"])
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.set_ylabel("Cost ($/M tokens)")
    ax.set_title("Operator Cost per Million\nUseful Tokens")

    # (1,1) Profit per hour across pricing tiers
    ax = axes[1, 1]
    tier_colors = {
        "premium": _COLORS["purple"],
        "standard": _COLORS["blue"],
        "budget": _COLORS["green"],
        "economy": _COLORS["red"],
    }
    bar_w = 0.8 / len(tiers)
    for j, tier in enumerate(tiers):
        profits = []
        for m in metrics_list:
            ec = m.get("econ", {})
            profits.append(ec.get(f"profit_{tier}", 0))
        offsets = x + (j - len(tiers) / 2 + 0.5) * bar_w
        bars = ax.bar(offsets, profits, bar_w,
                      color=tier_colors.get(tier, _COLORS["gray"]),
                      alpha=0.8, label=f"${econ.pricing_tiers[tier]}/M ({tier})")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.set_ylabel("Profit ($/hour)")
    ax.set_title("Hourly Profit by Pricing Tier")
    ax.legend(fontsize=_FS["legend"] - 1, loc="best", ncol=2)

    fig.suptitle("Real-World Economic Analysis", fontsize=_FS["title"] + 1,
                 fontweight="bold", y=1.01)

    # Caption with delta
    if len(metrics_list) >= 2:
        fcfs_ec = metrics_list[0].get("econ", {})
        lines = []
        for m in metrics_list[1:]:
            ec = m.get("econ", {})
            cost_delta = ec.get("total_cost_per_hour", 0) - fcfs_ec.get("total_cost_per_hour", 0)
            tput_delta = ec.get("throughput_tok_per_s", 0) - fcfs_ec.get("throughput_tok_per_s", 0)
            lines.append(
                f"{m['policy']}: cost {cost_delta:+.4f} $/hr, "
                f"throughput {tput_delta:+.0f} tok/s"
            )
        _add_caption(fig,
                     "vs FCFS: " + " | ".join(lines) +
                     "\nGPU amortization dominates cost; "
                     "energy savings matter most at economy pricing tier.",
                     y=-0.04)

    _savefig(fig, os.path.join(output_dir, "scheduling_economics.png"))


def _plot_summary_table(policies, metrics_list, output_dir, colors_map,
                        energy_model, econ):
    """Generate concept + summary table figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={"wspace": 0.3,
                                                 "width_ratios": [1, 1.3]})

    # Left: Scheduling concept — energy curve with zones
    pos_range = np.arange(0, 1001)
    energy_curve = np.array([energy_model(p) for p in pos_range])

    ax1.plot(pos_range, energy_curve, color=_COLORS["blue"], linewidth=2,
             label=f"E(p) = {energy_model.alpha:.0f} + {energy_model.beta:.3f}p")
    ax1.axhline(np.mean(energy_curve), color=_COLORS["gray"], linewidth=1,
                linestyle=":", label=f"Mean = {np.mean(energy_curve):.0f} mJ")

    # Highlight zones
    late_start = 700
    ax1.fill_between(pos_range[late_start:], energy_curve[late_start:] * 0,
                     energy_curve[late_start:],
                     alpha=0.12, color=_COLORS["red"],
                     label="PB: pause these (expensive)")
    ax1.fill_between(pos_range[:150], energy_curve[:150] * 0,
                     energy_curve[:150],
                     alpha=0.12, color=_COLORS["green"],
                     label="PB: admit new reqs (cheap)")

    ax1.annotate("New request:\nposition 0, cheapest",
                 xy=(50, energy_model(50)),
                 xytext=(200, energy_model(50) + 50),
                 fontsize=_FS["annotation"],
                 arrowprops=dict(arrowstyle="->", color=_COLORS["green"], lw=1),
                 color=_COLORS["green"])
    ax1.annotate("Old request:\nhigh position, expensive\n→ pause for a few steps",
                 xy=(850, energy_model(850)),
                 xytext=(500, energy_model(850) + 40),
                 fontsize=_FS["annotation"],
                 arrowprops=dict(arrowstyle="->", color=_COLORS["red"], lw=1),
                 color=_COLORS["red"])

    ax1.set_xlabel("Token Position in Sequence")
    ax1.set_ylabel("Energy per Token (mJ)")
    ax1.set_title("PB Scheduling Intuition")
    ax1.legend(loc="upper left", fontsize=_FS["legend"] - 0.5)

    # Right: Summary table
    ax2.axis("off")
    fcfs_m = metrics_list[0]

    table_data = [["Metric"] + policies]
    row_specs = [
        ("Completed", "completed", "{:.0f}", False),
        ("Abandoned", "abandoned", "{:.0f}", False),
        ("Useful tokens", "useful_tokens", "{:.0f}", False),
        ("Total energy (J)", "total_energy_J", "{:.1f}", True),
        ("Wasted energy (J)", "wasted_energy_J", "{:.1f}", True),
        ("mJ/useful tok", "energy_per_useful_token_mJ", "{:.1f}", True),
        ("Peak step (mJ)", "peak_step_energy_mJ", "{:.0f}", True),
        ("Avg step (mJ)", "avg_step_energy_mJ", "{:.0f}", True),
        ("Avg TTFT (steps)", "avg_ttft_steps", "{:.1f}", True),
        ("P95 TTFT", "p95_ttft_steps", "{:.1f}", True),
        ("Avg batch size", "avg_batch_size", "{:.1f}", False),
        ("Avg pos sum (×1k)", "avg_position_sum", "{:.0f}", True),
    ]
    delta_keys = {"total_energy_J", "wasted_energy_J", "energy_per_useful_token_mJ",
                  "peak_step_energy_mJ", "avg_step_energy_mJ",
                  "avg_ttft_steps", "avg_position_sum"}
    for label, key, fmt, show_delta in row_specs:
        row = [label]
        for m in metrics_list:
            val_raw = m[key]
            if key == "avg_position_sum":
                val_raw = val_raw / 1000  # display in thousands
            val = fmt.format(val_raw)
            if show_delta and m["policy"] != "FCFS":
                ref_raw = fcfs_m[key]
                if key == "avg_position_sum":
                    ref_raw = ref_raw / 1000
                if ref_raw > 0:
                    pct = (val_raw - ref_raw) / ref_raw * 100
                    val += f"\n({pct:+.1f}%)"
            row.append(val)
        table_data.append(row)

    # Add economic rows
    if econ and all("econ" in m for m in metrics_list):
        table_data.append(["--- Economics ---"] + [""] * len(policies))
        econ_rows = [
            ("Throughput (tok/s)", "throughput_tok_per_s", "{:.0f}", True),
            ("Cost ($/hr)", "total_cost_per_hour", "{:.4f}", True),
            ("Cost ($/M tok)", "cost_per_M_tokens", "{:.4f}", True),
        ]
        for tier in econ.pricing_tiers:
            econ_rows.append(
                (f"Profit-{tier} ($/hr)", f"profit_{tier}", "{:.3f}", True))
        for label, key, fmt, show_delta in econ_rows:
            row = [label]
            for m in metrics_list:
                ec = m.get("econ", {})
                val_raw = ec.get(key, 0)
                val = fmt.format(val_raw)
                if show_delta and m["policy"] != "FCFS":
                    ref = metrics_list[0].get("econ", {}).get(key, 0)
                    if ref != 0:
                        pct = (val_raw - ref) / abs(ref) * 100
                        val += f"\n({pct:+.1f}%)"
                row.append(val)
            table_data.append(row)

    table = ax2.table(cellText=table_data[1:],
                      colLabels=table_data[0],
                      loc="center",
                      cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.3)

    # Style header row
    for j in range(len(policies) + 1):
        cell = table[0, j]
        cell.set_facecolor("#E8E8E8")
        cell.set_text_props(fontweight="bold")

    # Highlight economics separator
    for j in range(len(policies) + 1):
        for i, row in enumerate(table_data[1:], start=1):
            if row[0].startswith("---"):
                cell = table[i, j]
                cell.set_facecolor("#F0F0F0")
                cell.set_text_props(fontstyle="italic")

    ax2.set_title("Summary Comparison", fontsize=_FS["title"],
                  fontweight="bold", pad=15)

    _add_caption(fig,
                 "PB (Position-Budget): proactively caps step energy by pausing "
                 "high-position requests. Reduces peak power and average energy, "
                 "trades throughput for cost savings. "
                 "Economics assume V100 GPU at $0.80/hr amortized.",
                 y=-0.04)

    _savefig(fig, os.path.join(output_dir, "scheduling_summary.png"))


# ═══════════════════════════════════════════════════════════
#  Multi-Scenario Sweep
# ═══════════════════════════════════════════════════════════

def run_load_sweep(energy_model: EnergyModel, econ: EconomicModel,
                   output_dir: str, max_batch: int = 60,
                   max_steps: int = 5000, cost_cap_factor: float = 0.85,
                   user_patience: int = 300, seed: int = 42,
                   seq_length_mean: int = 500, seq_length_std: int = 300):
    """
    Run simulations at multiple load levels and generate comparison.
    """
    # Load levels: fraction of max capacity
    # With batch=60, seq_mean=500: capacity ≈ 60/500 = 0.12 req/step
    capacity = max_batch / seq_length_mean
    load_levels = {
        "low (50%)":     capacity * 0.50,
        "medium (80%)":  capacity * 0.80,
        "high (100%)":   capacity * 1.00,
        "overload (120%)": capacity * 1.20,
    }

    all_results = {}
    all_metrics = {}

    for load_name, rate in load_levels.items():
        print(f"\n{'='*60}")
        print(f"  Load scenario: {load_name} (arrival_rate={rate:.4f})")
        print(f"{'='*60}")

        scenario_results = {}
        scenario_metrics = {}
        for policy in ["FCFS", "PAP", "PB"]:
            print(f"  Running {policy}...", end=" ", flush=True)
            r = run_simulation(
                policy_name=policy,
                energy_model=energy_model,
                arrival_rate=rate,
                max_batch=max_batch,
                max_steps=max_steps,
                seq_length_mean=seq_length_mean,
                seq_length_std=seq_length_std,
                cost_cap_factor=cost_cap_factor,
                user_patience=user_patience,
                seed=seed)
            m = compute_metrics(r, econ)
            scenario_results[policy] = r
            scenario_metrics[policy] = m
            print(f"completed={m['completed']}, "
                  f"energy={m['total_energy_J']:.0f}J, "
                  f"TTFT={m['avg_ttft_steps']:.1f}")

        all_results[load_name] = scenario_results
        all_metrics[load_name] = scenario_metrics

    # Generate sweep comparison figure
    _plot_load_sweep(all_metrics, load_levels, output_dir, econ)
    return all_results, all_metrics


def _plot_load_sweep(all_metrics, load_levels, output_dir, econ):
    """Generate the multi-load comparison figure."""
    os.makedirs(output_dir, exist_ok=True)
    load_names = list(load_levels.keys())
    policies = ["FCFS", "PAP", "PB"]
    colors_map = {
        "FCFS": _COLORS["blue"],
        "PAP": _COLORS["orange"],
        "PB": _COLORS["green"],
    }

    fig, axes = plt.subplots(2, 3, figsize=(12, 7),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.35})

    # Metric specifications: (key, ylabel, higher_is_better)
    metric_specs = [
        ("avg_step_energy_mJ", "Avg Step Energy (mJ)", axes[0, 0], False),
        ("peak_step_energy_mJ", "Peak Step Energy (mJ)", axes[0, 1], False),
        ("energy_per_useful_token_mJ", "mJ / Useful Token", axes[0, 2], False),
        ("completed", "Completed Requests", axes[1, 0], True),
        ("avg_ttft_steps", "Avg TTFT (steps)", axes[1, 1], False),
        ("abandoned", "Abandoned Requests", axes[1, 2], False),
    ]

    x = np.arange(len(load_names))
    bar_w = 0.8 / len(policies)

    for metric_key, ylabel, ax, higher_better in metric_specs:
        for j, policy in enumerate(policies):
            vals = []
            for load_name in load_names:
                m = all_metrics[load_name][policy]
                vals.append(m[metric_key])
            offsets = x + (j - len(policies) / 2 + 0.5) * bar_w
            ax.bar(offsets, vals, bar_w,
                   color=colors_map[policy], alpha=0.85,
                   edgecolor="white", linewidth=0.3,
                   label=policy if metric_key == metric_specs[0][0] else "")
        ax.set_xticks(x)
        ax.set_xticklabels([n.split("(")[1].rstrip(")") for n in load_names],
                           fontsize=_FS["tick"] - 0.5)
        ax.set_ylabel(ylabel, fontsize=_FS["label"] - 1)
        ax.set_xlabel("Load Level", fontsize=_FS["label"] - 1)

    # Add legend to first plot
    axes[0, 0].legend(fontsize=_FS["legend"])

    fig.suptitle("Scheduling Policy Comparison Across Load Levels",
                 fontsize=_FS["title"] + 1, fontweight="bold", y=1.01)

    _add_caption(fig,
                 "PB shows increasing advantage at higher loads: lower peak/avg energy, "
                 "fewer abandonments vs EB. PAP maintains throughput better than PB.",
                 y=-0.02)

    _savefig(fig, os.path.join(output_dir, "scheduling_load_sweep.png"))

    # Also generate economic sweep figure
    _plot_economic_sweep(all_metrics, load_levels, output_dir, econ)


def _plot_economic_sweep(all_metrics, load_levels, output_dir, econ):
    """Show profit difference (PB vs FCFS) across loads and pricing tiers."""
    if not econ:
        return

    load_names = list(load_levels.keys())
    tiers = list(econ.pricing_tiers.keys())
    tier_colors = {
        "premium": _COLORS["purple"],
        "standard": _COLORS["blue"],
        "budget": _COLORS["green"],
        "economy": _COLORS["red"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5),
                             gridspec_kw={"wspace": 0.35})

    x = np.arange(len(load_names))
    bar_w = 0.8 / len(tiers)

    # Left: PB profit delta vs FCFS
    ax = axes[0]
    for j, tier in enumerate(tiers):
        deltas = []
        for load_name in load_names:
            pb_ec = all_metrics[load_name]["PB"].get("econ", {})
            fcfs_ec = all_metrics[load_name]["FCFS"].get("econ", {})
            delta = pb_ec.get(f"profit_{tier}", 0) - fcfs_ec.get(f"profit_{tier}", 0)
            deltas.append(delta)
        offsets = x + (j - len(tiers) / 2 + 0.5) * bar_w
        ax.bar(offsets, deltas, bar_w,
               color=tier_colors.get(tier, _COLORS["gray"]),
               alpha=0.8, label=f"${econ.pricing_tiers[tier]}/M ({tier})")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([n.split("(")[1].rstrip(")") for n in load_names],
                       fontsize=_FS["tick"] - 0.5)
    ax.set_ylabel("Δ Profit ($/hour)\nPB − FCFS")
    ax.set_xlabel("Load Level")
    ax.set_title("PB Profit Impact vs FCFS\nby Pricing Tier")
    ax.legend(fontsize=_FS["legend"] - 1, ncol=2, loc="best")

    # Right: Energy savings (%) vs throughput loss (%)
    ax = axes[1]
    for load_name in load_names:
        fcfs = all_metrics[load_name]["FCFS"]
        pb = all_metrics[load_name]["PB"]
        if fcfs["avg_step_energy_mJ"] > 0:
            energy_save = (1 - pb["avg_step_energy_mJ"] / fcfs["avg_step_energy_mJ"]) * 100
        else:
            energy_save = 0
        if fcfs["useful_tokens"] > 0:
            tput_loss = (1 - pb["useful_tokens"] / fcfs["useful_tokens"]) * 100
        else:
            tput_loss = 0
        short_name = load_name.split("(")[1].rstrip(")")
        ax.scatter(tput_loss, energy_save, s=120, zorder=5,
                   color=_COLORS["blue"], edgecolors="white", linewidth=1)
        ax.annotate(short_name, (tput_loss, energy_save),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=_FS["annotation"])

    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Throughput Loss (%)\n(fewer useful tokens)")
    ax.set_ylabel("Energy Savings (%)\n(lower avg step energy)")
    ax.set_title("PB Trade-off:\nEnergy Saved vs Throughput Lost")

    # Add quadrant labels
    ax.text(0.95, 0.95, "Win-win\n(rare)", transform=ax.transAxes,
            ha="right", va="top", fontsize=_FS["annotation"],
            color=_COLORS["green"], alpha=0.6)
    ax.text(0.95, 0.05, "Lose-lose", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=_FS["annotation"],
            color=_COLORS["red"], alpha=0.6)

    fig.suptitle("Position-Budget: Economic Trade-offs",
                 fontsize=_FS["title"] + 1, fontweight="bold", y=1.02)

    _add_caption(fig,
                 "Key question: does PB's energy saving offset its throughput cost? "
                 "Answer depends on pricing tier — economy tier benefits most "
                 "because energy is a larger fraction of total cost.",
                 y=-0.04)

    _savefig(fig, os.path.join(output_dir, "scheduling_economic_tradeoff.png"))


# ═══════════════════════════════════════════════════════════
#  Print Summary
# ═══════════════════════════════════════════════════════════

def print_summary(results: dict, econ: EconomicModel = None):
    policies = list(results.keys())
    metrics_list = [compute_metrics(results[n], econ) for n in policies]

    print(f"\n{'='*78}")
    print(f"  SCHEDULING POLICY SIMULATION RESULTS")
    print(f"{'='*78}")

    fcfs_m = metrics_list[0]

    header = f"  {'Metric':<32s}"
    for m in metrics_list:
        header += f"  {m['policy']:>14s}"
    print(header)
    print(f"  {'-'*32}" + f"  {'-'*14}" * len(metrics_list))

    rows = [
        ("Completed requests", "completed", "{:.0f}", False),
        ("Abandoned requests", "abandoned", "{:.0f}", False),
        ("Useful tokens (completed)", "useful_tokens", "{:.0f}", False),
        ("Total tokens (incl. wasted)", "total_tokens", "{:.0f}", False),
        ("Total energy (J)", "total_energy_J", "{:.1f}", True),
        ("Wasted energy (J)", "wasted_energy_J", "{:.1f}", True),
        ("mJ / useful token", "energy_per_useful_token_mJ", "{:.2f}", True),
        ("Avg step energy (mJ)", "avg_step_energy_mJ", "{:.0f}", True),
        ("Peak step energy (mJ)", "peak_step_energy_mJ", "{:.0f}", True),
        ("P99 step energy (mJ)", "p99_step_energy_mJ", "{:.0f}", True),
        ("Avg batch size", "avg_batch_size", "{:.1f}", False),
        ("Batch utilization (%)", "batch_utilization", "{:.1f}", False),
        ("Avg position sum", "avg_position_sum", "{:.0f}", True),
        ("Avg TTFT (steps)", "avg_ttft_steps", "{:.1f}", True),
        ("Median TTFT", "median_ttft_steps", "{:.1f}", False),
        ("P95 TTFT", "p95_ttft_steps", "{:.1f}", True),
        ("Avg paused steps", "avg_paused_steps", "{:.1f}", False),
        ("Max paused steps", "max_paused_steps", "{:.0f}", False),
    ]

    for label, key, fmt, show_delta in rows:
        line = f"  {label:<32s}"
        for m in metrics_list:
            val = fmt.format(m[key])
            if show_delta and m["policy"] != "FCFS":
                ref = fcfs_m[key]
                if ref > 0:
                    pct = (m[key] - ref) / ref * 100
                    val += f" ({pct:+.1f}%)"
            line += f"  {val:>14s}"
        print(line)

    # Economic summary
    if econ:
        print(f"\n  {'─ Economics ─':<32s}" + f"  {'─'*14}" * len(metrics_list))
        for m in metrics_list:
            ec = m.get("econ", {})
            if not ec:
                continue

        econ_rows = [
            ("Throughput (tok/s)", "throughput_tok_per_s", "{:.0f}", True),
            ("Total cost ($/hr)", "total_cost_per_hour", "${:.4f}", True),
            ("  └ GPU amortization", "gpu_amort_per_hour", "${:.4f}", False),
            ("  └ Electricity", "electricity_per_hour", "${:.6f}", True),
            ("Cost per M tokens", "cost_per_M_tokens", "${:.4f}", True),
        ]
        for label, key, fmt, show_delta in econ_rows:
            line = f"  {label:<32s}"
            for m in metrics_list:
                ec = m.get("econ", {})
                val = fmt.format(ec.get(key, 0))
                if show_delta and m["policy"] != "FCFS":
                    ref = metrics_list[0].get("econ", {}).get(key, 0)
                    if ref != 0:
                        pct = (ec.get(key, 0) - ref) / abs(ref) * 100
                        val += f" ({pct:+.1f}%)"
                line += f"  {val:>14s}"
            print(line)

        # Profit per tier
        print()
        for tier, price in econ.pricing_tiers.items():
            line = f"  Profit @${price}/M ({tier:<8s})"
            for m in metrics_list:
                ec = m.get("econ", {})
                profit = ec.get(f"profit_{tier}", 0)
                margin = ec.get(f"margin_{tier}", 0)
                val = f"${profit:.3f}/hr"
                if m["policy"] != "FCFS":
                    ref_profit = metrics_list[0].get("econ", {}).get(f"profit_{tier}", 0)
                    delta = profit - ref_profit
                    val += f" ({delta:+.3f})"
                line += f"  {val:>14s}"
            print(line)

    print(f"\n{'='*78}")

    # Key findings
    for i, m in enumerate(metrics_list[1:], 1):
        e_save = (1 - m["avg_step_energy_mJ"] / fcfs_m["avg_step_energy_mJ"]) * 100 if fcfs_m["avg_step_energy_mJ"] > 0 else 0
        peak_save = (1 - m["peak_step_energy_mJ"] / fcfs_m["peak_step_energy_mJ"]) * 100 if fcfs_m["peak_step_energy_mJ"] > 0 else 0
        ttft_change = m["avg_ttft_steps"] - fcfs_m["avg_ttft_steps"]
        aband_diff = m["abandoned"] - fcfs_m["abandoned"]
        tput_diff = m["useful_tokens"] - fcfs_m["useful_tokens"]

        print(f"\n  Key findings ({m['policy']} vs FCFS):")
        print(f"    ⚡ Avg step energy:    {e_save:+.1f}%")
        print(f"    ⚡ Peak step energy:   {peak_save:+.1f}%")
        print(f"    📊 Useful tokens:      {tput_diff:+.0f} ({tput_diff/max(1,fcfs_m['useful_tokens'])*100:+.1f}%)")
        print(f"    ⏱  Avg TTFT:           {ttft_change:+.1f} steps")
        print(f"    ❌ Abandoned:           {aband_diff:+.0f}")

        if econ:
            ec = m.get("econ", {})
            fcfs_ec = fcfs_m.get("econ", {})
            cost_delta = ec.get("total_cost_per_hour", 0) - fcfs_ec.get("total_cost_per_hour", 0)
            print(f"    💰 Operator cost:      {cost_delta:+.4f} $/hr")
            for tier in ["economy", "budget", "standard", "premium"]:
                p_delta = ec.get(f"profit_{tier}", 0) - fcfs_ec.get(f"profit_{tier}", 0)
                print(f"    💰 Profit@{tier:<8s}:  {p_delta:+.4f} $/hr")

    print()


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Simulate position-aware scheduling policies "
                    "with real-world economic analysis")
    parser.add_argument("--data_csv", type=str, default=None,
                        help="Path to token_energy_stream_per_position.csv "
                             "to calibrate energy model from real data")
    parser.add_argument("--output_dir", type=str, default="output/scheduling_sim",
                        help="Directory to save output figures")
    parser.add_argument("--arrival_rate", type=float, default=None,
                        help="Mean request arrivals per step (Poisson). "
                             "If unset, runs multi-load sweep.")
    parser.add_argument("--max_batch", type=int, default=60,
                        help="Maximum batch size (max_num_seqs)")
    parser.add_argument("--max_steps", type=int, default=5000,
                        help="Simulation duration in steps")
    parser.add_argument("--seq_length_mean", type=int, default=500,
                        help="Mean output sequence length")
    parser.add_argument("--seq_length_std", type=int, default=300,
                        help="Std of output sequence length")
    parser.add_argument("--cost_cap_factor", type=float, default=0.85,
                        help="PB cost cap = factor × max_batch × E(mean_pos). "
                             "Lower = more aggressive energy capping.")
    parser.add_argument("--user_patience", type=int, default=300,
                        help="Steps without token before user abandons request")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    # Economic parameters
    parser.add_argument("--gpu_cost_per_hour", type=float, default=0.80,
                        help="GPU amortized cost ($/hour)")
    parser.add_argument("--electricity_per_kwh", type=float, default=0.10,
                        help="Electricity cost ($/kWh)")
    parser.add_argument("--step_duration_s", type=float, default=0.030,
                        help="Duration per decode step (seconds)")
    args = parser.parse_args()

    # Load or create energy model
    if args.data_csv and os.path.exists(args.data_csv):
        print(f"📊 Loading energy model from: {args.data_csv}")
        energy_model = EnergyModel.from_csv(args.data_csv)
    else:
        print("📊 Using default energy model (calibrated from Qwen3-8B stream data)")
        energy_model = EnergyModel()

    print(f"   E(p) = {energy_model.alpha:.1f} + {energy_model.beta:.4f} × p  mJ")
    print(f"   E(0) = {energy_model.alpha:.0f} mJ,  "
          f"E(250) = {energy_model(250):.0f} mJ,  "
          f"E(500) = {energy_model(500):.0f} mJ")
    print(f"   Ratio E(500)/E(0) = {energy_model(500)/energy_model.alpha:.2f}x\n")

    # Create economic model
    econ = EconomicModel(
        gpu_cost_per_hour=args.gpu_cost_per_hour,
        electricity_per_kwh=args.electricity_per_kwh,
        step_duration_s=args.step_duration_s,
    )

    print(f"💰 Economic model:")
    print(f"   GPU cost:       ${econ.gpu_cost_per_hour:.2f}/hr (amortized)")
    print(f"   Electricity:    ${econ.electricity_per_kwh:.2f}/kWh")
    print(f"   Step duration:  {econ.step_duration_s*1000:.0f} ms")
    print(f"   Pricing tiers:  {econ.pricing_tiers}")
    print()

    if args.arrival_rate is not None:
        # Single-scenario run
        print(f"⚙️  Single-scenario simulation:")
        print(f"   arrival_rate = {args.arrival_rate} req/step")
        print(f"   max_batch = {args.max_batch}")
        print(f"   max_steps = {args.max_steps}")
        print(f"   seq_length ~ N({args.seq_length_mean}, {args.seq_length_std})")
        print(f"   cost_cap_factor = {args.cost_cap_factor}")
        print(f"   user_patience = {args.user_patience}")
        print(f"   seed = {args.seed}\n")

        common_kwargs = dict(
            energy_model=energy_model,
            arrival_rate=args.arrival_rate,
            max_batch=args.max_batch,
            max_steps=args.max_steps,
            seq_length_mean=args.seq_length_mean,
            seq_length_std=args.seq_length_std,
            cost_cap_factor=args.cost_cap_factor,
            user_patience=args.user_patience,
            seed=args.seed,
        )

        results = {}
        for policy in ["FCFS", "PAP", "PB"]:
            print(f"🔄 Running {policy}...", end=" ", flush=True)
            r = run_simulation(policy_name=policy, **common_kwargs)
            results[policy] = r
            m = compute_metrics(r, econ)
            print(f"done. {m['completed']} completed, "
                  f"{m['total_energy_J']:.0f} J, "
                  f"TTFT={m['avg_ttft_steps']:.1f}")

        print_summary(results, econ)

        print(f"\n📈 Generating visualizations...")
        os.makedirs(args.output_dir, exist_ok=True)
        plot_comparison(results, args.output_dir, energy_model, econ)
        print(f"\n✅ All figures saved to {args.output_dir}/")

    else:
        # Multi-load sweep
        print(f"⚙️  Multi-load sweep:")
        print(f"   max_batch = {args.max_batch}")
        print(f"   max_steps = {args.max_steps}")
        print(f"   seq_length ~ N({args.seq_length_mean}, {args.seq_length_std})")
        print(f"   cost_cap_factor = {args.cost_cap_factor}")
        print(f"   user_patience = {args.user_patience}\n")

        all_results, all_metrics = run_load_sweep(
            energy_model=energy_model,
            econ=econ,
            output_dir=args.output_dir,
            max_batch=args.max_batch,
            max_steps=args.max_steps,
            cost_cap_factor=args.cost_cap_factor,
            user_patience=args.user_patience,
            seed=args.seed,
            seq_length_mean=args.seq_length_mean,
            seq_length_std=args.seq_length_std,
        )

        # Print summary for each scenario
        for load_name in all_results:
            print(f"\n\n{'#'*78}")
            print(f"  SCENARIO: {load_name}")
            print(f"{'#'*78}")
            print_summary(all_results[load_name], econ)

        print(f"\n📈 Generating sweep visualizations...")
        print(f"\n✅ All figures saved to {args.output_dir}/")

    # Save metrics as JSON for reproducibility
    json_path = os.path.join(args.output_dir, "scheduling_metrics.json")
    os.makedirs(args.output_dir, exist_ok=True)
    if args.arrival_rate is not None:
        metrics_json = {p: compute_metrics(results[p], econ)
                        for p in results}
        # Remove non-serializable items
        for p in metrics_json:
            if "econ" in metrics_json[p]:
                metrics_json[p]["econ"] = {
                    k: v for k, v in metrics_json[p]["econ"].items()
                    if isinstance(v, (int, float, str))}
        with open(json_path, "w") as f:
            json.dump(metrics_json, f, indent=2, default=str)
        print(f"📝 Metrics saved to {json_path}")


if __name__ == "__main__":
    main()
