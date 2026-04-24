"""Generate benchmark and training-curve charts for the CatanRL report."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ─────────────────────────────────────────────
# 1. BENCHMARK BAR CHART
# ─────────────────────────────────────────────
opponents  = ["vs WeightedRandom ×3", "vs ValueFunction\n+ WR ×2", "vs AlphaBeta\n+ WR ×2"]
baseline   = [49.3,  2.0,  1.0]
teammate   = [64.0, 13.0,  7.0]
self_play  = [64.8, 10.5,  6.0]
random_ch  = 25.0

x     = np.arange(len(opponents))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5))

b1 = ax.bar(x - width, baseline,  width, label="Baseline (WR-trained)",         color="#5B9BD5", edgecolor="white", linewidth=0.5)
b2 = ax.bar(x,          teammate,  width, label="Teammate (vp_ablation_stage2)", color="#ED7D31", edgecolor="white", linewidth=0.5)
b3 = ax.bar(x + width,  self_play, width, label="Self-Play (self_play_4p)",      color="#70AD47", edgecolor="white", linewidth=0.5)

ax.axhline(random_ch, color="#C00000", linestyle="--", linewidth=1.3,
           label=f"Random chance ({random_ch:.0f}%)")

for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

ax.set_ylabel("Win Rate (%)", fontsize=11)
ax.set_title("CatanRL Model Benchmark — 4-Player Catan", fontsize=13, pad=12)
ax.set_xticks(x)
ax.set_xticklabels(opponents, fontsize=10)
ax.set_ylim(0, 82)
ax.legend(fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig("charts/benchmark.png", bbox_inches="tight")
print("Saved: charts/benchmark.png")


# ─────────────────────────────────────────────
# 2. TRAINING CURVE  (vp_stage3 run)
# ─────────────────────────────────────────────
# Data extracted from terminal logs.
# steps are absolute; new steps = step - 27_002_880 (start of vp_stage3).
early_steps   = [27_012_881, 27_022_881, 27_032_881, 27_042_881]
early_recent  = [42, 48, 48, 54]
early_overall = [43.8, 44.3, 44.5, 45.4]

# Dense block from user-pasted logs (91.8% – 93.4%)
late_steps = [
    43_132_881, 43_142_881, 43_152_881, 43_162_881, 43_172_881,
    43_182_881, 43_192_881, 43_202_881, 43_212_881, 43_222_881,
    43_232_881, 43_242_881, 43_252_881, 43_262_881, 43_272_881,
    43_282_881, 43_292_881, 43_302_881, 43_312_881, 43_322_881,
    43_332_881, 43_342_881, 43_352_881, 43_362_881, 43_372_881,
    43_382_881, 43_392_881, 43_402_881, 43_412_881, 43_422_881,
    43_432_881, 43_442_881, 43_452_881, 43_462_881, 43_472_881,
    43_482_881, 43_492_881, 43_502_881, 43_512_881, 43_522_881,
    43_532_881, 43_542_881, 43_552_881, 43_562_881, 43_572_881,
    43_582_881, 43_592_881, 43_602_881, 43_612_881, 43_622_881,
    43_632_881, 43_642_881, 43_652_881, 43_662_881, 43_672_881,
    43_682_881, 43_692_881, 43_702_881, 43_712_881, 43_722_881,
    43_732_881, 43_742_881, 43_752_881, 43_762_881, 43_772_881,
    43_782_881, 43_792_881, 43_802_881, 43_812_881, 43_822_881,
    43_832_881, 43_842_881, 43_852_881, 43_862_881, 43_872_881,
    43_882_881,
]
late_recent = [
    72, 56, 68, 58, 60, 60, 58, 72, 64, 46,
    62, 52, 64, 48, 56, 66, 58, 62, 46, 74,
    56, 44, 54, 68, 68, 60, 60, 56, 54, 54,
    62, 62, 56, 60, 56, 68, 58, 66, 76, 64,
    60, 52, 64, 64, 50, 56, 62, 62, 60, 60,
    68, 56, 50, 52, 56, 64, 48, 66, 70, 62,
    54, 60, 60, 68, 68, 60, 68, 58, 48, 66,
    58, 60, 54, 70, 56, 54,
]
late_overall = [
    52.1, 52.1, 52.2, 52.2, 52.2, 52.2, 52.2, 52.2, 52.2, 52.2,
    52.2, 52.2, 52.2, 52.2, 52.2, 52.2, 52.2, 52.2, 52.2, 52.2,
    52.2, 52.2, 52.2, 52.2, 52.3, 52.3, 52.3, 52.3, 52.3, 52.3,
    52.3, 52.3, 52.3, 52.3, 52.3, 52.3, 52.3, 52.3, 52.3, 52.3,
    52.3, 52.3, 52.3, 52.3, 52.3, 52.3, 52.4, 52.4, 52.4, 52.4,
    52.4, 52.4, 52.4, 52.4, 52.4, 52.4, 52.4, 52.4, 52.4, 52.4,
    52.4, 52.4, 52.4, 52.4, 52.4, 52.4, 52.4, 52.4, 52.4, 52.5,
    52.5, 52.5, 52.5, 52.5, 52.5, 52.5,
]

peak_step   = 29_262_881
peak_recent = 78

all_steps   = early_steps   + late_steps
all_recent  = early_recent  + late_recent
all_overall = early_overall + late_overall

# Use absolute step numbers (in millions) so the peak at 29.26M sits
# visibly between the early cluster (~27M) and late cluster (~43M).
abs_steps = [s / 1e6 for s in all_steps]
peak_abs  = peak_step / 1e6

fig, ax = plt.subplots(figsize=(10, 5))

ax.scatter(abs_steps, all_recent, s=12, alpha=0.45, color="#5B9BD5", label="Win rate (last 50 eps)", zorder=2)
ax.plot(abs_steps, all_overall, color="#ED7D31", linewidth=2.2, label="Overall win rate", zorder=3)

ax.axhline(random_ch, color="#C00000", linestyle="--", linewidth=1.1,
           label=f"Random chance ({random_ch:.0f}%)")

ax.scatter([peak_abs], [peak_recent], s=120, color="#C55A11", zorder=5, marker="*")
ax.annotate(
    f"Peak: {peak_recent}%\n(step {peak_step:,})",
    xy=(peak_abs, peak_recent),
    xytext=(peak_abs + 1.5, peak_recent - 9),
    fontsize=8.5,
    arrowprops=dict(arrowstyle="->", color="black", lw=1),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.8),
)

ax.set_xlabel("Training Step (millions)", fontsize=11)
ax.set_ylabel("Win Rate (%)", fontsize=11)
ax.set_title("vp_stage3 Training Curve\n(vs 1 PPO-self + 2 VictoryPoint)", fontsize=13, pad=10)
ax.set_xlim(26.5, 44.5)
ax.set_ylim(0, 90)
ax.legend(fontsize=9, framealpha=0.9)

note = "Note: log data unavailable for steps 27.05 M – 43.1 M (gap in terminal output)"
ax.text(0.02, 0.03, note, transform=ax.transAxes, fontsize=7.5,
        color="gray", style="italic")

plt.tight_layout()
plt.savefig("charts/training_curve.png", bbox_inches="tight")
print("Saved: charts/training_curve.png")

plt.show()
