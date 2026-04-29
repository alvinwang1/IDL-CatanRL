# CatanRL: Sample-Efficient Reinforcement Learning for Settlers of Catan

This repository implements a deep reinforcement learning pipeline for the board game **Settlers of Catan**, built on the [Catanatron](https://github.com/bcollazo/catanatron) simulator. We use **MaskablePPO** to handle the game's large and dynamic action space, combined with reward shaping and curriculum learning to achieve competitive performance against heuristic bots.

## Key Features

- **MaskablePPO Implementation**: Uses `sb3-contrib` to zero out invalid actions (e.g., building without resources), significantly accelerating convergence.
- **Reward Shaping (PBRS)**: Implements Potential-Based Reward Shaping using a normalized board-value heuristic.
- **Phi-Cap Mechanism**: A custom solution to the "terminal shaping shock" problem, ensuring that terminal rewards remain positive and stable.
- **GNN Encoder**: A heterogeneous Graph Neural Network that models Catan's board topology (nodes, edges, and tiles) via message passing.
- **Population-Based Self-Play**: A snapshot-pool mechanism that allows the agent to train against its own past versions to improve robustness.
- **Curriculum Training**: A multi-stage pipeline that progressively introduces harder opponents and varying win conditions (e.g., the "8-VP bridge").

## Performance Highlights

| Agent | Opponents | Steps | Win Rate |
|---|---|---|---|
| Sparse Baseline | 3× WeightedRandom | 1M | 53.0% |
| PBRS + Curriculum | 3× WeightedRandom | 27M | **64.0%** |
| Self-Play Fine-Tuning | 3× WeightedRandom | 32M | **64.8%** |
| PBRS + Curriculum | 1× ValueFunction | 27M | **10.7%** |

*Note: Random chance in a 4-player game is 25%.*

## Project Structure

- `catanrl/`: Core package containing environment wrappers, reward functions, and model architectures.
  - `models.py`: MLP and GNN (CatanGraphEncoder) implementations.
  - `reward.py`: PBRS, VP-Delta, and Phi-Cap reward logic.
  - `self_play.py`: Population-based snapshot pool callback.
- `configs/`: YAML files for reproducible training runs and curricula.
- `report/`: LaTeX source for the project paper.
- `notebooks/`: Jupyter notebooks for baselines and performance visualization.

## Setup & Training

### 1. Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Training with Curriculum
To launch a full curriculum training run using a GNN encoder:
```bash
python -m catanrl.train --config configs/gnn_pbrs_4p.yaml
```

### 3. Resuming Training
To resume from a specific stage:
```bash
python -m catanrl.train --config configs/gnn_resume_stage3.yaml
```

## Acknowledgments
This project was developed for **11-785 Introduction to Deep Learning** at Carnegie Mellon University. We build upon the excellent work of [Bruno Collazo](https://github.com/bcollazo) on Catanatron.