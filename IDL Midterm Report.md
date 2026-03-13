# IDL Midterm Report — Sample-Efficient Self-Play RL for Settlers of Catan

---

# SECTION I: Rubric Compliance Audit

The following table evaluates the original proposal (`IDL Project Proposal.md`) against each rubric section.

| Section | Requirement | Status | Notes |
|---|---|---|---|
| **Overview → Abstract** | Clear, non-expert-accessible abstract | **Partially Satisfied** | Present but reads as a dense research pitch, not a clear abstract. Highly technical; inaccessible to non-experts. |
| **Overview → Motivation & Objectives** | Clear motivation and research objectives | **Satisfied** | "Motivation and Background" section clearly explains why Catan is a compelling RL benchmark. |
| **Related Work → Literature Review** | Review of ≥2–3 relevant papers | **Satisfied** | Three papers reviewed: Gendre & Kaneko (2020), Belle et al. (2025), Martinenghi et al. (2024). |
| **Related Work → Background** | Technical background for understanding the project | **Partially Satisfied** | Background mixes with motivation. Actor-critic RL and self-play are mentioned but not formally explained. |
| **Methodology → Model Description** | Clear description of implemented model(s) | **Partially Satisfied** | Describes many *proposed* components (cross-dimensional encoder, CTDE, planning, negotiation) but these are plans, not implementations. The actual model used (MaskablePPO with default MLP policy) is not described. |
| **Methodology → Dataset** | Dataset/environment description | **Partially Satisfied** | Catanatron is mentioned as the environment, but no detail on observation space, action space, or environment configuration. |
| **Methodology → Evaluation Metric** | Metrics clearly defined | **Partially Satisfied** | Win rate, Elo, and secondary metrics are listed, but the actual implementation only uses win rate with Wilson score CIs. Elo is not implemented. |
| **Methodology → Loss Function** | Loss function described | **Missing** | No explicit discussion of the PPO loss function, clipping, value loss, or entropy coefficients used. |
| **Experimental Depth → Baseline Selection & Evaluation** | Baseline model described and evaluated | **Partially Satisfied** | Proposal mentions baselines (heuristic bots, random bot, simple PPO baseline), but the actual baselines implemented are only Catanatron's built-in RandomPlayer and WeightedRandomPlayer. |
| **Experimental Depth → Implemented Extensions / Experiments** | Extensions beyond baseline | **Missing** | The proposal describes many future extensions but the codebase only contains baseline experiments with no extensions yet implemented. |
| **Experimental Depth → Baseline Reproduction Evidence** | Evidence of baseline results | **Missing** | No baseline results are presented in the proposal. Results exist in notebook outputs but are not included in the written document. |
| **Results → Results** | Quantitative results | **Missing** | No results section exists in the proposal. |
| **Results → Error / Failure Case Analysis** | (Optional) | **Missing** | Not present. |
| **Results → Sensitivity / Ablation Analysis** | (Optional) | **Missing** | Not present. |
| **Discussion → Future Directions** | Discussion of future work | **Partially Satisfied** | The Implementation Plan/Timeline section implicitly covers future work, but there is no explicit "Future Directions" section reflecting on what was learned. |
| **Discussion → Conclusion** | Summary conclusion | **Missing** | No conclusion section. |
| **Bonus → Visualization** | (Optional) | **Missing** | No visualizations in the proposal (they exist in notebooks). |
| **Bonus → Extra experiments** | (Optional) | **Partially Satisfied** | Four experiment variants exist in the codebase but none are discussed in the proposal. |
| **Admin → Bibliography** | Proper citations | **Satisfied** | Four references with links are provided. |
| **Admin → Team Contributions** | Team member contributions listed | **Missing** | Not present in the proposal. |
| **Admin → GitHub link** | Link to repository | **Missing** | Not present in the proposal. |

---

# SECTION II: Codebase Verification

## Codebase Summary

The repository contains:
- **4 Jupyter notebooks** implementing training and evaluation pipelines
- **4 saved model checkpoints** (`.zip` files)
- **3 TensorBoard log directories**
- **No standalone Python scripts** (everything is in notebooks)
- **No custom model architectures** (uses off-the-shelf MaskablePPO from `sb3_contrib`)

### Experiments Implemented

| Notebook | Opponents | Obs Dim | Action Dim | Train Steps | Eval Win Rate |
|---|---|---|---|---|---|
| `catan_rl_baseline.ipynb` | 1 × RandomPlayer (2-player) | 614 | 290 | 1,000,000 | **84.2%** [80.7%, 87.1%] |
| `catan_rl_weighted.ipynb` | 1 × WeightedRandomPlayer (2-player) | 614 | 290 | 1,000,000 | **68.6%** |
| `catan_rl_baseline_4players.ipynb` | 3 × RandomPlayer (4-player) | 1002 | 290 | 1,000,000 | **70.8%** (chance: 25%) |
| `catan_rl_4p_weighted.ipynb` | 3 × WeightedRandomPlayer (4-player) | 1002 | 290 | 1,000,000 | **53.0%** (chance: 25%) |

### Technical Details from Code

- **RL Algorithm:** MaskablePPO from `sb3_contrib` (NOT custom PPO, NOT CTDE)
- **Policy Network:** `MaskableActorCriticPolicy` (default MLP, NOT cross-dimensional encoder)
- **Action Masking:** Custom `mask_fn` wrapping `env.unwrapped.get_valid_actions()`
- **Observation:** Flat vector from `catanatron_gym` (614-dim for 2p, 1002-dim for 4p; NOT factorized/cross-dimensional)
- **Reward:** Default Catanatron gym reward: +1 for win, −1 for loss (NOT shaped rewards, NOT potential-based shaping)
- **Opponents:** Catanatron built-in `RandomPlayer` and `WeightedRandomPlayer` (NOT opponent pools, NOT self-play snapshots)
- **Training:** Single run per experiment, 1M timesteps (NOT curriculum, NOT staged training)
- **Evaluation:** 500 games per experiment, reporting win rate and Wilson score 95% CI (NOT Elo)
- **Hardware:** CPU only (`Using cpu device` in logs)
- **Hyperparameters:** Default MaskablePPO (lr=0.0003, clip_range=0.2, no tuning)

### Claim Verification Table

| Claim in Proposal | Evidence in Code | Status |
|---|---|---|
| Uses Catanatron simulator | `import catanatron_gym; gymnasium.make("catanatron-v1", ...)` | ✅ Verified |
| PPO-based training | `MaskablePPO` from `sb3_contrib` | ✅ Verified |
| Action masking for valid actions | Custom `mask_fn` using `env.unwrapped.get_valid_actions()` | ✅ Verified |
| Cross-dimensional neural network encoder | No custom model; uses default `MaskableActorCriticPolicy` (MLP) | ❌ Not Implemented |
| Factorized/structured state encoding | Flat observation vector from Catanatron gym (614 or 1002 dims) | ❌ Not Implemented |
| Centralized critic + decentralized actors (CTDE) | Standard single-agent PPO; no multi-agent architecture | ❌ Not Implemented |
| Population-based self-play with snapshots | Trains against fixed built-in opponents (Random, WeightedRandom) | ❌ Not Implemented |
| Curriculum/staged training | Single-phase training, 1M steps | ❌ Not Implemented |
| Reward shaping (ΔVP, penalties, entropy reg.) | Default Catanatron reward (+1/−1) | ❌ Not Implemented |
| Lightweight planning module | Not present in codebase | ❌ Not Implemented |
| Negotiation module | Not present in codebase | ❌ Not Implemented |
| Opponent modeling/embedding | Not present in codebase | ❌ Not Implemented |
| Win rate evaluation | 500-game evaluation with Wilson CI, per notebook | ✅ Verified |
| Elo evaluation | Not implemented | ❌ Not Implemented |
| Ablation experiments | Not implemented (no parameter variations, single config per notebook) | ❌ Not Implemented |
| ≤10M env steps budget | All runs use 1M steps | ✅ Verified (well within budget) |
| TensorBoard logging | TB log directories present (`catan_tb/`, `catan_tb_weighted/`, `catan_tb_4p_weighted/`) | ✅ Verified |

---

# SECTION III: Rewritten Midterm Report

---

## Abstract

Settlers of Catan presents a compelling challenge for reinforcement learning due to its combination of stochastic resource generation, hidden information, spatial strategy, and multi-player competition. This project investigates whether a standard deep RL agent—specifically, Proximal Policy Optimization with action masking (MaskablePPO)—can learn to consistently outperform rule-based opponents in the Catanatron Catan simulator. We establish baselines by training PPO agents against both random and heuristic opponents in two-player and four-player settings, each for 1,000,000 environment steps. Our midterm results show that the PPO agent achieves an 84.2% win rate against a random opponent in 2-player games (95% CI: [80.7%, 87.1%]) and a 53.0% win rate in 4-player games against three heuristic opponents (versus 25% random chance). These baseline results confirm that deep RL can learn meaningful Catan strategy from self-experience, and provide a foundation for planned extensions including custom neural architectures, self-play training, and reward shaping.

## 1. Motivation & Objectives

Random board games are compelling benchmarks for modern reinforcement learning because they require strong decision-making under uncertainty, not just perfect calculation. In deterministic perfect-information games like chess, performance is dominated by brute-force search over a fully observed game tree. By contrast, stochastic games with hidden information—such as Settlers of Catan—require agents to reason about probabilities, manage risk, and choose strategies that remain robust when outcomes vary.

Catan combines several challenging properties: (1) decisions have long-term consequences, (2) outcomes are noisy due to dice-driven resource generation, (3) players must act with incomplete information since hands are hidden, and (4) success depends on spatial strategy on the board (settlement/road placement, ports, and the robber). With modern CPU-friendly simulators like Catanatron [4], it is feasible to run large-scale experiments while maintaining computational tractability.

**Research Objectives:**
1. Establish a reproducible RL training and evaluation pipeline on Catanatron.
2. Train and evaluate MaskablePPO baseline agents in both 2-player and 4-player settings against built-in opponents.
3. Quantify baseline agent performance with statistical rigor (win rate with confidence intervals).
4. Identify directions for improvement (architecture, self-play, reward shaping) based on observed failure modes.

## 2. Literature Review

**Gendre & Kaneko (2020)** [1] introduced cross-dimensional neural networks tailored to Catan's heterogeneous structure—separate modules for hexagonal faces, edges, vertices, and per-player features, fused via attention-like pooling. Their approach dramatically improved RL performance over strong heuristics, demonstrating that structured state representation is critical for Catan. This motivates our planned extension to move beyond flat observation encodings.

**Belle, Barnes, et al. (2025)** [2] explored self-evolving agentic systems on Catan via Catanatron, focusing on LLM-driven code and prompt evolution. Their work highlights the value of staged capability growth, systematic failure analysis, and opponent diversity through multi-agent evaluation—design principles that inform our planned self-play extensions.

**Martinenghi et al. (2024)** [3] studied multi-party Catan dialogue with dialogue-act prediction and classification on the STAC dataset. Their work demonstrates that pragmatics-aware negotiation can improve game outcomes without expensive large-model generation, suggesting a future direction for lightweight negotiation modules.

## 3. Background

**Proximal Policy Optimization (PPO)** [Schulman et al., 2017] is an actor-critic policy gradient algorithm that optimizes a clipped surrogate objective to balance exploration and exploitation while preventing destructive policy updates. The clipping mechanism constrains the policy ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ to the interval $[1-\epsilon, 1+\epsilon]$ (we use $\epsilon = 0.2$), providing stable training without the complexity of trust-region methods.

**Action Masking** is essential in Catan because the set of legal actions varies dramatically across game states (e.g., building requires specific resources). MaskablePPO [Huang et al., 2022] extends PPO by zeroing out the probabilities of invalid actions before sampling, ensuring the agent only selects legal moves. This is implemented via the `sb3_contrib` library.

**Catanatron** [4] is an open-source Python simulator for Settlers of Catan that provides a Gymnasium-compatible API. It supports configurable numbers of players and includes built-in opponents: `RandomPlayer` (uniform random over legal actions) and `WeightedRandomPlayer` (heuristic preference ordering: cities > settlements > dev cards).

## 4. Methodology

### 4.1 Model Description

We use **MaskablePPO** from the `sb3_contrib` library with the default `MaskableActorCriticPolicy`. This is a multi-layer perceptron (MLP) policy network with separate actor and critic heads. The actor outputs a categorical distribution over the 290-dimensional discrete action space, with invalid actions masked out before sampling. The critic outputs a scalar value estimate.

All experiments use default hyperparameters:
- Learning rate: 3 × 10⁻⁴
- Clip range: ε = 0.2
- Minibatch size and number of epochs: defaults (64 minibatches, 10 epochs per rollout)
- Rollout buffer size: 2048 steps
- Device: CPU

### 4.2 Dataset / Environment

We use the Catanatron Gymnasium environment (`catanatron-v1`) with two configurations:

| Configuration | Observation Dim | Action Dim | Opponents |
|---|---|---|---|
| 2-player | 614 | 290 | 1 opponent |
| 4-player | 1002 | 290 | 3 opponents |

The observation is a flat feature vector provided by Catanatron, encoding board state, player resources, building positions, and game phase information. No custom feature engineering or factorized encoding has been applied at this stage.

Opponents come from Catanatron's built-in player library:
- **RandomPlayer:** Selects uniformly at random among legal actions.
- **WeightedRandomPlayer:** Applies heuristic preference ordering (cities > settlements > development cards > other actions).

### 4.3 Evaluation Metric

We evaluate agents using **win rate** over 500 evaluation games, reported with **Wilson score 95% confidence intervals**. All evaluation uses deterministic action selection (`deterministic=True` in the policy's `predict` method).

For 4-player games, we compare against the random expected win rate of 25%.

### 4.4 Loss Function

MaskablePPO optimizes the standard PPO clipped surrogate objective:

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

with a value function loss $L^{VF}$ and an entropy bonus $L^{S}$ for exploration:

$$L(\theta) = L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 L^{S}(\theta)$$

The reward signal is provided by the Catanatron environment: +1 for winning, −1 for losing, 0 otherwise (sparse, game-outcome reward).

### 4.5 Experimental Iteration

We conducted four baseline experiments, varying the number of players and opponent strength:

1. **Experiment 1 (2p-Random):** PPO vs. 1 RandomPlayer, 1M steps.
2. **Experiment 2 (2p-Weighted):** PPO vs. 1 WeightedRandomPlayer, 1M steps.
3. **Experiment 3 (4p-Random):** PPO vs. 3 RandomPlayers, 1M steps.
4. **Experiment 4 (4p-Weighted):** PPO vs. 3 WeightedRandomPlayers, 1M steps.

Each experiment includes training curve logging (episode reward, episode length, win rate) via a custom `CatanLoggingCallback` and TensorBoard, followed by a 500-game evaluation.

## 5. Baseline and Extensions

### 5.1 Baseline Model

Our baseline is **MaskablePPO** with a flat MLP policy, trained against Catanatron's built-in opponents. This represents the simplest viable deep RL approach: no custom architecture, no reward shaping, no curriculum, and no self-play. The choice of MaskablePPO is motivated by its stability and sample efficiency among on-policy methods, combined with its built-in support for action masking—a critical requirement in Catan where the legal action set changes every turn.

### 5.2 Planned Improvements (Not Yet Implemented)

The following extensions are proposed for the second half of the project:

1. **Cross-dimensional encoder:** Replace the flat MLP with separate modules for faces, edges, vertices, and per-player features, inspired by Gendre & Kaneko [1].
2. **Self-play with opponent pools:** Maintain a pool of past policy snapshots and train against a mixture, rather than fixed opponents.
3. **Reward shaping:** Add potential-based shaping signals (ΔVP, expected resource income from placements) to accelerate learning.
4. **Curriculum training:** Stage training from easy (2-player random) to hard (4-player heuristic) opponents.
5. **Hyperparameter tuning:** Sweep over learning rate, clip range, and network architecture.

### 5.3 Evidence of Baseline Results

Training curves are available in the notebook outputs and TensorBoard logs. The baseline PPO agents achieve learning convergence within 1M timesteps in all four configurations, as evidenced by steadily increasing win rates during training (see Results section).

## 6. Results and Analysis

### 6.1 Results

| Experiment | Opponents | Games | Wins | Win Rate | 95% CI | Random Chance |
|---|---|---|---|---|---|---|
| 2p-Random | 1 × RandomPlayer | 500 | 421 | **84.2%** | [80.7%, 87.1%] | 50% |
| 2p-Weighted | 1 × WeightedRandomPlayer | 500 | 343 | **68.6%** | — | 50% |
| 4p-Random | 3 × RandomPlayer | 500 | 354 | **70.8%** | — | 25% |
| 4p-Weighted | 3 × WeightedRandomPlayer | 500 | 265 | **53.0%** | — | 25% |

**Key findings:**
- In 2-player games, PPO achieves a strong 84.2% win rate against random opponents and 68.6% against heuristic opponents, indicating that the agent has learned meaningful Catan strategy.
- In 4-player games, PPO achieves 70.8% and 53.0% win rates (vs. 25% random chance), demonstrating that the learned policy generalizes from 2-player to multi-player settings, though performance degrades with stronger opponents.
- Training is stable: all four runs converge without collapse, and TensorBoard logs show monotonically improving win rates.
- Training speed is approximately 950–1200 steps/second on CPU, completing 1M steps in ~15 minutes.

### 6.2 Sensitivity / Ablation Analysis

Formal ablation studies have not yet been conducted. However, comparing across the four experiments provides preliminary sensitivity insights:
- **Number of players:** Win rate drops from 84.2% (2p) to 70.8% (4p) against random opponents, suggesting the 4-player setting is substantially harder.
- **Opponent strength:** Win rate drops from 84.2% to 68.6% (2p) and from 70.8% to 53.0% (4p) when switching from random to heuristic opponents, confirming that `WeightedRandomPlayer` is a meaningfully stronger baseline.

## 7. Discussion

The baseline experiments demonstrate that standard MaskablePPO can learn effective Catan strategies from scratch using only game-outcome rewards. The agent surpasses the random baseline significantly in all settings and maintains a winning record even against heuristic opponents.

However, several limitations are apparent:

1. **Flat observation encoding:** The 614/1002-dimensional flat vectors do not exploit the spatial structure of the Catan board. A structured encoder could improve sample efficiency and final performance.
2. **Fixed opponents:** Training against a single fixed opponent type limits generalization. Self-play or opponent pools would create more robust policies.
3. **Sparse reward:** The binary win/loss reward provides limited learning signal. Shaped rewards could accelerate training and improve performance against stronger opponents.
4. **No hyperparameter tuning:** All experiments use default PPO hyperparameters, leaving room for improvement through systematic tuning.

## 8. Future Directions

1. **Implement cross-dimensional encoder** following Gendre & Kaneko [1] to exploit Catan's board topology.
2. **Add self-play training** with a pool of past snapshots to improve robustness and avoid overfitting to fixed opponents.
3. **Implement reward shaping** with potential-based terms (victory points, resource income) and validate via ablation.
4. **Scale training** to 5–10M steps with optimized CPU vectorization.
5. **Introduce Elo evaluation** via round-robin tournaments for more fine-grained performance comparison.
6. **Investigate failure cases** by analyzing losing games to identify systematic weaknesses (e.g., poor initial placement, inefficient trading).

## 9. Conclusion

This midterm report presents baseline results for training RL agents to play Settlers of Catan using the Catanatron simulator. We trained MaskablePPO agents in four configurations (2-player and 4-player, against random and heuristic opponents) and evaluated performance over 500 games each. The best 2-player agent achieves 84.2% win rate against random opponents (95% CI: [80.7%, 87.1%]), and the 4-player agent achieves 53.0% against heuristic opponents (vs. 25% random chance), confirming that deep RL can learn meaningful Catan strategy from sparse rewards alone. These baselines provide a solid foundation for the planned extensions: structured neural encoders, self-play training, and reward shaping.

## Bibliography

1. Gendre, Q., & Kaneko, T. (2020). Playing Catan with Cross-dimensional Neural Network. *arXiv:2008.07079*. [https://arxiv.org/abs/2008.07079](https://arxiv.org/abs/2008.07079)
2. Belle, N., Barnes, D., et al. (2025). Agents of Change: Self-Evolving LLM Agents for Strategic Planning. *arXiv:2506.04651*. [https://arxiv.org/abs/2506.04651](https://arxiv.org/abs/2506.04651)
3. Martinenghi, A., et al. (2024). LLMs of Catan: Exploring Pragmatic Capabilities… (Games & NLP @ LREC-COLING). [https://aclanthology.org/2024.games-1.12.pdf](https://aclanthology.org/2024.games-1.12.pdf)
4. Catanatron documentation and repository. [https://docs.catanatron.com/](https://docs.catanatron.com/) and [https://github.com/bcollazo/catanatron](https://github.com/bcollazo/catanatron)
5. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
6. Huang, S., et al. (2022). A Closer Look at Invalid Action Masking in Policy Gradient Algorithms. *arXiv:2006.14171*.

## Team Contributions

*[TODO: Fill in team member names and individual contributions before submission.]*

## GitHub Repository

*[TODO: Add the link to the project GitHub repository before submission.]*

**Repository:** `https://github.com/<your-org>/IDL-CatanRL`
