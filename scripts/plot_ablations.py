"""
Generate report-quality figures from ablation TensorBoard logs.

Usage:
    python scripts/plot_ablations.py
    python scripts/plot_ablations.py --logdir logs/ablation --outdir figures

Produces:
  figures/learning_curves_lambda.pdf   — win rate vs steps, lambda sweep
  figures/learning_curves_lr.pdf       — win rate vs steps, LR schedule comparison
  figures/bar_final_winrate.pdf        — bar chart of final win rates
  figures/shaping_vs_winrate.pdf       — PBRS shaping magnitude vs win rate (scatter)
"""

import argparse
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ── Condition metadata ──────────────────────────────────────────────────────

CONDITIONS = [
    # (log_subdir, display_label, group, linestyle, color)
    ("lam0_const",   r"Sparse ($\lambda=0$), const LR",   "lambda", "-",  "#555555"),
    ("lam01_const",  r"PBRS $\lambda=0.1$, const LR",     "lambda", "-",  "#2196F3"),
    ("lam05_const",  r"PBRS $\lambda=0.5$, const LR",     "lambda", "-",  "#4CAF50"),
    ("lam10_const",  r"PBRS $\lambda=1.0$, const LR",     "lambda", "-",  "#F44336"),
    ("lam0_linear",  r"Sparse ($\lambda=0$), linear LR",  "lr",     "--", "#555555"),
    ("lam05_linear", r"PBRS $\lambda=0.5$, linear LR",    "lr",     "--", "#4CAF50"),
]

RANDOM_CHANCE = 0.25  # 4-player game


# ── TensorBoard reader ───────────────────────────────────────────────────────

def load_tb_scalar(log_dir, tag):
    """Return (steps, values) arrays from the most recent TB event file in log_dir."""
    # SB3 creates MaskablePPO_N sub-dirs; pick the latest
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
    """Exponential moving average smoothing (same as TensorBoard default)."""
    smoothed = []
    last = values[0] if len(values) else 0.0
    for v in values:
        last = last * weight + (1 - weight) * v
        smoothed.append(last)
    return np.array(smoothed)


# ── Shared plot style ────────────────────────────────────────────────────────

FIGSIZE_WIDE = (8, 4.5)
FIGSIZE_SQ   = (6, 5)

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


# ── Figure 1: Learning curves — lambda sweep ─────────────────────────────────

def plot_lambda_curves(logdir, outdir):
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    lambda_conditions = [c for c in CONDITIONS if c[2] == "lambda"]

    for subdir, label, _, ls, color in lambda_conditions:
        path = os.path.join(logdir, subdir)
        steps, vals = load_tb_scalar(path, "catan/win_rate_50ep")
        if len(steps) == 0:
            print(f"  [WARN] No data for {subdir}, skipping")
            continue
        steps_m = steps / 1e6
        ax.plot(steps_m, smooth(vals), label=label, linestyle=ls, color=color, linewidth=2)
        # Shaded raw variance
        ax.fill_between(steps_m, smooth(vals, 0.7), smooth(vals, 0.95),
                        color=color, alpha=0.10)

    ax.axhline(RANDOM_CHANCE, color="black", linewidth=1, linestyle=":", alpha=0.6,
               label="Random baseline (25%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Win rate (50-ep rolling avg)")
    ax.set_title(r"Effect of PBRS $\lambda$ on learning speed")
    ax.legend(loc="upper left", framealpha=0.85)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    out = os.path.join(outdir, "learning_curves_lambda.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    print(f"  Saved: {out}")
    plt.close(fig)


# ── Figure 2: Learning curves — LR schedule comparison ───────────────────────

def plot_lr_curves(logdir, outdir):
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    # Show the two sparse and two lam0.5 conditions side by side
    lr_conditions = [
        c for c in CONDITIONS
        if c[0] in ("lam0_const", "lam0_linear", "lam05_const", "lam05_linear")
    ]

    for subdir, label, _, ls, color in lr_conditions:
        path = os.path.join(logdir, subdir)
        steps, vals = load_tb_scalar(path, "catan/win_rate_50ep")
        if len(steps) == 0:
            print(f"  [WARN] No data for {subdir}, skipping")
            continue
        steps_m = steps / 1e6
        ax.plot(steps_m, smooth(vals), label=label, linestyle=ls, color=color, linewidth=2)

    ax.axhline(RANDOM_CHANCE, color="black", linewidth=1, linestyle=":", alpha=0.6,
               label="Random baseline (25%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Win rate (50-ep rolling avg)")
    ax.set_title("Effect of LR schedule (constant vs linear decay)")
    ax.legend(loc="upper left", framealpha=0.85)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    out = os.path.join(outdir, "learning_curves_lr.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    print(f"  Saved: {out}")
    plt.close(fig)


# ── Figure 3: Bar chart — final win rates ────────────────────────────────────

def plot_bar_final(logdir, outdir):
    labels, win_rates, colors = [], [], []

    for subdir, label, _, _, color in CONDITIONS:
        path = os.path.join(logdir, subdir)
        steps, vals = load_tb_scalar(path, "catan/win_rate_50ep")
        if len(vals) == 0:
            print(f"  [WARN] No data for {subdir}, skipping bar")
            continue
        # Final win rate = mean of last 5 logged points (~last 50k steps)
        final_wr = float(np.mean(vals[-5:]))
        labels.append(label)
        win_rates.append(final_wr)
        colors.append(color)

    if not labels:
        print("  [WARN] No data for bar chart")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.3), 4.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, win_rates, color=colors, width=0.55, edgecolor="white", linewidth=0.8)
    ax.axhline(RANDOM_CHANCE, color="black", linewidth=1.2, linestyle="--", alpha=0.7,
               label=f"Random baseline ({RANDOM_CHANCE*100:.0f}%)")

    # Value labels on bars
    for bar, wr in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{wr*100:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Win rate (final 50 episodes)")
    ax.set_title("Final win rate by ablation condition")
    ax.legend(framealpha=0.85)
    ax.set_ylim(0, min(1.0, max(win_rates) * 1.25 + 0.05))
    fig.tight_layout()

    out = os.path.join(outdir, "bar_final_winrate.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    print(f"  Saved: {out}")
    plt.close(fig)


# ── Figure 4: Shaping magnitude vs win rate (PBRS diagnostic) ────────────────

def plot_shaping_vs_winrate(logdir, outdir):
    """Scatter: mean PBRS shaping per episode vs final win rate, one point per condition."""
    pbrs_conditions = [c for c in CONDITIONS if "lam0" not in c[0] or "lam0_" not in c[0]]
    # All conditions that have pbrs shaping data (non-zero lambda)
    pbrs_conditions = [
        c for c in CONDITIONS
        if c[0] not in ("lam0_const", "lam0_linear")
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)

    for subdir, label, _, _, color in pbrs_conditions:
        path = os.path.join(logdir, subdir)
        _, shaping = load_tb_scalar(path, "catan/pbrs_shaping_per_ep")
        _, wr = load_tb_scalar(path, "catan/win_rate_50ep")
        if len(shaping) == 0 or len(wr) == 0:
            continue
        # Use last-quarter means for stability
        q = max(1, len(shaping) // 4)
        mean_shaping = float(np.mean(shaping[-q:]))
        final_wr = float(np.mean(wr[-5:]))
        ax.scatter(mean_shaping, final_wr, color=color, s=100, zorder=3)
        ax.annotate(label, (mean_shaping, final_wr),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8.5, color=color)

    ax.axhline(RANDOM_CHANCE, color="black", linewidth=1, linestyle=":", alpha=0.6,
               label="Random baseline")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel(r"Mean PBRS shaping per episode ($\lambda \cdot \sum_t \Delta\Phi$)")
    ax.set_ylabel("Final win rate")
    ax.set_title("PBRS shaping magnitude vs. win rate")
    ax.legend(framealpha=0.85)
    fig.tight_layout()

    out = os.path.join(outdir, "shaping_vs_winrate.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    print(f"  Saved: {out}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logs/ablation",
                        help="Directory containing per-condition TB logs")
    parser.add_argument("--outdir", default="figures",
                        help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Reading logs from: {args.logdir}")
    print(f"Saving figures to: {args.outdir}\n")

    plot_lambda_curves(args.logdir, args.outdir)
    plot_lr_curves(args.logdir, args.outdir)
    plot_bar_final(args.logdir, args.outdir)
    plot_shaping_vs_winrate(args.logdir, args.outdir)

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()
