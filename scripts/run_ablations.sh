#!/usr/bin/env bash
# Run all ablation conditions sequentially.
# Usage: bash scripts/run_ablations.sh
# Logs to logs/ablation/<condition>/ and saves models to models/ablation/<condition>.zip
# After all runs complete, evaluate each and append results to ablation_results.txt

set -e
cd "$(dirname "$0")/.."

mkdir -p models/ablation logs/ablation

CONFIGS=(
  "configs/ablation/lam0_const.yaml"
  "configs/ablation/lam01_const.yaml"
  "configs/ablation/lam05_const.yaml"
  "configs/ablation/lam10_const.yaml"
  "configs/ablation/lam0_linear.yaml"
  "configs/ablation/lam05_linear.yaml"
)

LABELS=(
  "sparse_const"
  "lam0.1_const"
  "lam0.5_const"
  "lam1.0_const"
  "sparse_linear"
  "lam0.5_linear"
)

MODELS=(
  "models/ablation/lam0_const"
  "models/ablation/lam01_const"
  "models/ablation/lam05_const"
  "models/ablation/lam10_const"
  "models/ablation/lam0_linear"
  "models/ablation/lam05_linear"
)

RESULTS_FILE="ablation_results.txt"
echo "Ablation results — $(date)" > "$RESULTS_FILE"
echo "Condition | Win rate | 95% CI" >> "$RESULTS_FILE"
echo "---------|----------|-------" >> "$RESULTS_FILE"

N=${#CONFIGS[@]}
for i in $(seq 0 $((N-1))); do
  CFG="${CONFIGS[$i]}"
  LABEL="${LABELS[$i]}"
  MODEL="${MODELS[$i]}"

  echo ""
  echo "=========================================="
  echo "  Run $((i+1))/$N: $LABEL"
  echo "  Config: $CFG"
  echo "=========================================="

  python -m catanrl.train --config "$CFG"

  echo ""
  echo "  Evaluating $LABEL..."
  python -m catanrl.evaluate \
    --config "$CFG" \
    --model "${MODEL}.zip" \
    | tee /tmp/eval_out.txt

  # Extract win rate line and append to results
  WR_LINE=$(grep "Win rate:" /tmp/eval_out.txt | head -1)
  CI_LINE=$(grep "95% CI:" /tmp/eval_out.txt | head -1)
  echo "$LABEL | $WR_LINE | $CI_LINE" >> "$RESULTS_FILE"

  echo ""
  echo "  Done: $LABEL"
done

echo ""
echo "=========================================="
echo "  ALL ABLATIONS COMPLETE"
echo "  Results saved to $RESULTS_FILE"
echo "  TensorBoard: tensorboard --logdir logs/ablation"
echo "=========================================="
cat "$RESULTS_FILE"
