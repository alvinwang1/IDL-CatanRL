"""
Generate report-quality figures for the VP shaping ablation and full training story.

Usage (after ablation runs):
    python scripts/plot_vp_ablation.py

After the overnight run, fill in EVAL_RESULTS below and re-run to get Figure 5.

Produces (all saved as both .pdf and .png in figures/):
  vp_ablation_curves.pdf   — win rate vs steps for each vp_scale condition
  vp_delta_signal.pdf      — VP delta reward received per episode vs steps
  vp_bar_final.pdf         — bar chart of final win rates across ablation conditions
  curriculum_progression.pdf — full training story with stage boundaries annotated
  opponent_ladder.pdf      — grouped bar chart: win rate vs WR and VF for key checkpoints
"""

import argparse
import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import FancyArrowPatch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ── Fill in after running evaluate.py on each checkpoint ─────────────────────
#
# Structure: { "Model label": {"WeightedRandom": 0.84, "ValueFunction": 0.06} }
# Leave as None to skip Figure 5 until you have the numbers.
#
# Example eval commands:
#   python -m catanrl.evaluate --config configs/vp_ablation.yaml \
#     --model models/overnight_vf_stage1.zip
#   python -m catanrl.evaluate --config configs/vp_ablation.yaml \
#     --model models/overnight_vf_stage1.zip \
#     --override "enemies=['ValueFunction','WeightedRandom','WeightedRandom']"

EVAL_RESULTS = {
    "WR-only model\n(lam05_linear_cont)":    {"WeightedRandom": 0.74,  "ValueFunction": 0.07},
    "VF-8 stage\n(vp_ablation_stage0)":       {"WeightedRandom": 0.595, "ValueFunction": 0.08},
    "VF-10 stage\n(vp_ablation_stage1)":      {"WeightedRandom": None,  "ValueFunction": 0.095},
    "Final model\n(vp_ablation, all stages)": {"WeightedRandom": 0.66,  "ValueFunction": 0.107},
}
# Note: None entries are skipped in the bar chart
# Original eval commands used to produce these numbers:
#   python -m catanrl.evaluate --config configs/vp_ablation.yaml --model <checkpoint>
#   --override "enemies=['ValueFunction','WeightedRandom','WeightedRandom']" --override eval_games=200

_EVAL_RESULTS_BACKUP = None  # Set this to a dict once you have numbers, e.g.:
# EVAL_RESULTS = {
#     "WR-only\n(ablation best)":   {"WeightedRandom": 0.84, "ValueFunction": 0.06},
#     "Stage 1 checkpoint\n(WR warmup)": {"WeightedRandom": 0.82, "ValueFunction": 0.05},
#     "Stage 2 checkpoint\n(VP bridge)": {"WeightedRandom": 0.80, "ValueFunction": 0.07},
#     "Final model\n(VP shaping)":  {"WeightedRandom": 0.75, "ValueFunction": 0.12},
# }


# ── Ablation conditions ────────────────────────────────────────────────────────

VP_CONDITIONS = [
    # (log_subdir, display_label, color)
    ("vp000", r"$\alpha=0.0$ (PBRS only, no VP reward)",     "#555555"),
    ("vp005", r"$\alpha=0.05$ (VP reward, conservative)",    "#2196F3"),
    ("vp010", r"$\alpha=0.1$  (VP reward, recommended)",     "#4CAF50"),
    ("vp020", r"$\alpha=0.2$  (VP reward, aggressive)",      "#F44336"),
]

# Curriculum stage boundaries for the overnight run (cumulative steps)
# Update these if your actual stage counts differ
CURRICULUM_STAGES = [
    (0,           "Start\n(Stage 1 ckpt)"),
    (500_000,     "Stage 1 end\n(WR warmup)"),
    (3_500_000,   "Stage 2 end\n(VP bridge)"),
    (11_500_000,  "Stage 3 end\n(VF-8)"),
    (25_000_000,  "Stage 4 end\n(VF-10)"),
    (30_500_000,  "Stage 5 end\n(2×VF)"),
]

RANDOM_CHANCE = 0.25   # 4-player game
VF_BASELINE_WR = 0.05  # rough previous best vs ValueFunction


# ── Shared helpers ─────────────────────────────────────────────────────────────

def load_tb_scalar(log_dir, tag):
    """Return (steps, values) from the most recent MaskablePPO_N sub-dir."""
    run_dirs = sorted(glob.glob(os.path.join(log_dir, "MaskablePPO_*")))
    if not run_dirs:
        run_dirs = [log_dir]
    event_dir = run_dirs[-1]
    ea = EventAccumulator(event_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return np.array([]), np.array([])
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def smooth(values, weight=0.85):
    """Exponential moving average (same as TensorBoard default)."""
    if len(values) == 0:
        return values
    smoothed, last = [], float(values[0])
    for v in values:
        last = last * weight + (1 - weight) * v
        smoothed.append(last)
    return np.array(smoothed)


def save_fig(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in (".pdf", ".png"):
        path = os.path.join(outdir, name + ext)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {os.path.join(outdir, name)}.pdf/.png")
    plt.close(fig)


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9.5,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ── Figure 1: VP ablation learning curves ─────────────────────────────────────

def plot_vp_curves(logdir, outdir):
    """
    Win rate (50-ep rolling average) vs training steps for each vp_scale value.
    All runs start from the same checkpoint (overnight_vf_stage1, after WR warmup
    and VictoryPoint bridge). The grey dashed line shows the previous best VF win
    rate (~5%) without VP shaping. A line that rises above this confirms VP rewards help.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    any_data = False

    for subdir, label, color in VP_CONDITIONS:
        path = os.path.join(logdir, subdir)
        steps, vals = load_tb_scalar(path, "catan/win_rate_50ep")
        if len(steps) == 0:
            print(f"  [WARN] No TB data for {subdir} — run not complete yet")
            continue
        any_data = True
        steps_m = steps / 1e6
        ax.plot(steps_m, smooth(vals), label=label, color=color, linewidth=2)
        ax.fill_between(steps_m, smooth(vals, 0.7), smooth(vals, 0.95),
                        color=color, alpha=0.08)

    if not any_data:
        print("  [SKIP] No data yet — run ablations first")
        plt.close(fig)
        return

    ax.axhline(RANDOM_CHANCE, color="black", linewidth=1, linestyle=":",
               alpha=0.5, label="Random chance (25%)")
    ax.axhline(VF_BASELINE_WR, color="#888888", linewidth=1.2, linestyle="--",
               alpha=0.7, label=f"Previous best vs VF (~{VF_BASELINE_WR*100:.0f}%)")

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel("Win rate — 50-episode rolling average")
    ax.set_title(
        r"VP delta reward ($\alpha$) ablation: effect on win rate vs ValueFunction"
        "\n"
        r"PBRS $\lambda=0.5$, $\phi_{cap}=4.0$, vps=8 | Same starting checkpoint for all runs"
    )
    ax.legend(loc="upper left", framealpha=0.85)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    save_fig(fig, outdir, "vp_ablation_curves")


# ── Figure 2: VP delta reward signal ──────────────────────────────────────────

def plot_vp_delta_signal(logdir, outdir):
    """
    Mean VP delta reward received per episode vs training steps, for each non-zero
    vp_scale condition. A healthy signal should be ~0.5–2.0 (winning agents rack up
    VPs; losing agents get less). If this is near zero for all conditions, the agent
    is not gaining VPs = not making progress toward winning.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    any_data = False

    for subdir, label, color in VP_CONDITIONS:
        if subdir == "vp000":
            continue  # no VP reward to show
        path = os.path.join(logdir, subdir)
        steps, vals = load_tb_scalar(path, "catan/vp_delta_per_ep")
        if len(steps) == 0:
            print(f"  [WARN] No vp_delta data for {subdir}")
            continue
        any_data = True
        steps_m = steps / 1e6
        ax.plot(steps_m, smooth(vals, 0.8), label=label, color=color, linewidth=2)

    if not any_data:
        print("  [SKIP] No VP delta data yet")
        plt.close(fig)
        return

    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel("Mean VP delta reward per episode")
    ax.set_title(
        "VP delta reward signal over training\n"
        "Rising trend = agent accumulating more VPs per game = learning to win"
    )
    ax.legend(loc="upper left", framealpha=0.85)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    save_fig(fig, outdir, "vp_delta_signal")


# ── Figure 3: Bar chart — final win rates ─────────────────────────────────────

def plot_vp_bar(logdir, outdir):
    """
    Final win rate (mean of last 5 TB log points ≈ last 250k steps) for each
    vp_scale condition. The grey bar (alpha=0) is the PBRS-only baseline from the
    same checkpoint. Any bar significantly above it validates VP shaping.
    """
    labels, win_rates, colors = [], [], []

    for subdir, label, color in VP_CONDITIONS:
        path = os.path.join(logdir, subdir)
        _, vals = load_tb_scalar(path, "catan/win_rate_50ep")
        if len(vals) == 0:
            continue
        labels.append(label)
        win_rates.append(float(np.mean(vals[-5:])))
        colors.append(color)

    if not labels:
        print("  [SKIP] No bar data yet")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.8), 4.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, win_rates, color=colors, width=0.55, edgecolor="white", linewidth=0.8)

    ax.axhline(RANDOM_CHANCE, color="black", linewidth=1.2, linestyle=":",
               alpha=0.5, label=f"Random chance ({RANDOM_CHANCE*100:.0f}%)")
    ax.axhline(VF_BASELINE_WR, color="#888888", linewidth=1.2, linestyle="--",
               alpha=0.7, label=f"Prev. best vs VF ({VF_BASELINE_WR*100:.0f}%)")

    for bar, wr in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{wr*100:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Win rate — final 50 episodes")
    ax.set_title(
        r"Final win rate by VP shaping strength ($\alpha$)"
        "\n"
        "3M steps vs ValueFunction (vps=8) from the same Stage 2 checkpoint"
    )
    ax.legend(framealpha=0.85)
    ax.set_ylim(0, min(1.0, max(win_rates + [VF_BASELINE_WR]) * 1.35 + 0.02))

    save_fig(fig, outdir, "vp_bar_final")


# ── Figure 4: Full curriculum progression ─────────────────────────────────────

def plot_curriculum_progression(overnight_logdir, outdir):
    """
    Win rate over the entire training history of the best overnight run, from
    WeightedRandom warm-up through VP bridge to ValueFunction curriculum stages.
    Vertical dashed lines mark curriculum stage boundaries. This is the headline
    figure showing the agent's complete learning trajectory.

    Pass --overnight_logdir logs/vp_ablation (or logs/overnight_vf_cont) after
    the overnight run completes.
    """
    steps, vals = load_tb_scalar(overnight_logdir, "catan/win_rate_50ep")
    if len(steps) == 0:
        print(f"  [SKIP] No data in {overnight_logdir} — overnight run not done yet")
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    steps_m = steps / 1e6
    ax.plot(steps_m, smooth(vals, 0.9), color="#1565C0", linewidth=2, label="Win rate (50-ep avg)")
    ax.fill_between(steps_m, smooth(vals, 0.7), smooth(vals, 0.97),
                    color="#1565C0", alpha=0.10)

    # Stage boundary annotations
    stage_colors = ["#BBDEFB", "#C8E6C9", "#FFE0B2", "#F8BBD9", "#E1BEE7"]
    prev_step_m = 0
    for i, (step, label) in enumerate(CURRICULUM_STAGES[1:], 0):
        step_m = step / 1e6
        ax.axvline(step_m, color="#BDBDBD", linewidth=1.2, linestyle="--", zorder=1)
        # Shade the stage region
        ax.axvspan(prev_step_m, step_m, alpha=0.06,
                   color=stage_colors[i % len(stage_colors)], zorder=0)
        # Stage label at top
        mid = (prev_step_m + step_m) / 2
        ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.95,
                label, ha="center", va="top", fontsize=7.5,
                color="#424242", transform=ax.get_xaxis_transform())
        prev_step_m = step_m

    ax.axhline(RANDOM_CHANCE, color="black", linewidth=1, linestyle=":",
               alpha=0.4, label="Random chance (25%)")

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Cumulative training steps (millions)")
    ax.set_ylabel("Win rate — 50-episode rolling average")
    ax.set_title(
        "Full training progression: WeightedRandom curriculum → ValueFunction\n"
        "Shaded regions = curriculum stages; opponent difficulty increases left to right"
    )
    ax.legend(loc="upper left", framealpha=0.85)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    save_fig(fig, outdir, "curriculum_progression")


# ── Figure 5: Opponent ladder ─────────────────────────────────────────────────

def plot_opponent_ladder(outdir):
    """
    Grouped bar chart showing win rate vs two opponent types (WeightedRandom and
    ValueFunction) for key training checkpoints. Reveals whether VF improvement
    came at the cost of WR performance (ideally it didn't). Fill in EVAL_RESULTS
    at the top of this file after running evaluate.py on each checkpoint.
    """
    if EVAL_RESULTS is None:
        print("  [SKIP] EVAL_RESULTS not filled in — run evaluate.py and update the dict")
        return

    models = list(EVAL_RESULTS.keys())
    opponent_types = ["WeightedRandom", "ValueFunction"]
    colors = {"WeightedRandom": "#42A5F5", "ValueFunction": "#EF5350"}

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 2), 5))

    for i, opp in enumerate(opponent_types):
        vals = [EVAL_RESULTS[m].get(opp) for m in models]
        offset = (i - 0.5) * width
        for j, (xpos, v) in enumerate(zip(x, vals)):
            if v is None:
                continue
            bar = ax.bar(xpos + offset, v, width, color=colors[opp],
                         edgecolor="white", linewidth=0.6,
                         label=opp if j == 0 else "_nolegend_")
            ax.text(xpos + offset,
                    v + 0.01,
                    f"{v*100:.0f}%", ha="center", va="bottom", fontsize=8.5)

    ax.axhline(RANDOM_CHANCE, color="black", linewidth=1, linestyle=":",
               alpha=0.5, label="Random chance (25%)")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Win rate (300 evaluation games)")
    ax.set_title(
        "Win rate vs WeightedRandom and ValueFunction — key checkpoints\n"
        "Blue = WeightedRandom opponents (easier), Red = ValueFunction opponents (harder)"
    )
    ax.legend(framealpha=0.85)
    ax.set_ylim(0, 1.05)

    save_fig(fig, outdir, "opponent_ladder")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate VP shaping report figures")
    parser.add_argument("--logdir", default="logs/ablation/vp",
                        help="Directory containing vp000/vp005/vp010/vp020 TB logs")
    parser.add_argument("--overnight_logdir", default=None,
                        help="TB log dir for the full overnight run (for Figure 4). "
                             "E.g. logs/vp_ablation or logs/overnight_vf_cont")
    parser.add_argument("--outdir", default="figures",
                        help="Output directory for saved figures")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Ablation logs:  {args.logdir}")
    print(f"Output dir:     {args.outdir}\n")

    print("Figure 1: VP ablation learning curves")
    plot_vp_curves(args.logdir, args.outdir)

    print("Figure 2: VP delta reward signal")
    plot_vp_delta_signal(args.logdir, args.outdir)

    print("Figure 3: Bar chart — final win rates")
    plot_vp_bar(args.logdir, args.outdir)

    if args.overnight_logdir:
        print("Figure 4: Full curriculum progression")
        plot_curriculum_progression(args.overnight_logdir, args.outdir)
    else:
        print("Figure 4: [SKIPPED] Pass --overnight_logdir after the overnight run completes")

    print("Figure 5: Opponent ladder")
    plot_opponent_ladder(args.outdir)

    print("\nDone. All figures saved as .pdf and .png")


if __name__ == "__main__":
    main()
