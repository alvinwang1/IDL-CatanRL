import sys

if len(sys.argv) > 1:
    print("Usage: python -m catanrl.train --config <path>")
    print("       python -m catanrl.evaluate --config <path> --model <path>")
else:
    print("CatanRL - RL agent for Settlers of Catan")
    print()
    print("Commands:")
    print("  python -m catanrl.train --config configs/pbrs_4p_weighted.yaml")
    print("  python -m catanrl.evaluate --config configs/pbrs_4p_weighted.yaml --model models/pbrs_4p_weighted.zip")
    print()
    print("Lambda ablation sweep:")
    print("  for lam in 0.0 0.1 0.5 1.0; do")
    print("    python -m catanrl.train --config configs/pbrs_4p_weighted.yaml --override pbrs_lambda=$lam --override save_path=models/ablation_lam_$lam")
    print("  done")
