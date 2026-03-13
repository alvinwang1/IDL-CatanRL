**Sample-Efficient Self-Play RL for Settlers of Catan with Cross-Dimensional Representation and Lightweight Planning**

**Abstract:**  
This project trains a compute-bounded Catan RL agent (≤10M env steps/experiment; ≤2 weeks/run) and targets statistically significant improvements in win rate and Elo over strong heuristic and learning baselines. The approach uses a compact, structured state encoding of board topology and player inventories, sample-efficient self-play, and limited lookahead/planning applied only to high-impact sparse decisions (initial placement; pivotal build/trade turns). 

To mitigate multi-agent non-stationarity, we use population-based self-play with opponent mixing and periodic evaluation against frozen policy snapshots. Optionally, we add a compute-light negotiation module based on dialogue-act (intent) prediction rather than free-form generation. Evaluation emphasizes generalization across unseen board seeds and opponent mixtures, learning stability, and ablations isolating the contribution of each component. Deliverables include a reproducible training/evaluation pipeline, a trained agent, a baseline suite, a full empirical study, and an open-source release suitable for a workshop submission.

**Motivation and Background**  
Random board games are compelling benchmarks for modern reinforcement learning because they require strong decision-making under uncertainty, not just perfect calculation. In deterministic perfect-information games like chess or checkers, optimal play is mostly about searching a known game tree from a fully observed state, and performance can be dominated by brute-force lookahead plus value evaluation. 

By contrast, stochastic games introduce randomness (e.g., dice or card draws) and often hidden information, so agents must learn to reason about probabilities, manage risk, and choose strategies that remain robust when outcomes vary. This makes them better proxies for real-world settings, where actions have delayed effects, data is incomplete, and the same decision can lead to different results.

Catan is a strong setting for publishable multi-agent RL because it combines several hard, realistic challenges: (1) decisions have long-term consequences, (2) outcomes are noisy due to dice-driven resource generation, (3) and players must act with incomplete information since hands are hidden. 

Success also depends on spatial strategy on the board (settlement/road placement, ports, and the robber) and on trading/negotiation, not just tactical move selection. Unlike perfect-information games, strong play requires good state representation and training methods that remain stable against changing opponents, especially because the initial placement phase effectively commits players to a strategy for much of the game. With modern CPU-friendly simulators like Catanatron, it’s feasible to run large-scale experiments and still keep the study rigorous under tight compute budgets.

**Related Work (modern \+ classic; connect to design choices)**  
Gendre & Kaneko (2020) 

- ross-dimensional neural networks tailored to Catan’s heterogeneous structure (faces/edges/vertices \+ per-player features \+ multi-head outputs) and show dramatic RL improvement  
- outperforms strong heuristics ([arXiv](https://arxiv.org/abs/2008.07079?utm_source=chatgpt.com))

“Agents of Change” (Belle et al., 2025\) 

- self-evolving agentic systems on Catan via Catanatron   
- focus on LLM-driven code/prompt evolution  
- staged capability growth, systematic failure analysis, and opponent diversity via multi-agent evaluation ([arXiv](https://arxiv.org/abs/2506.04651?utm_source=chatgpt.com))

“LLMs of Catan” (Martinenghi et al., 2024\) 

- multi-party Catan dialogue with dialogue-act prediction/classification on the STAC dataset  
- highlights challenges in speaker/turn structure and pragmatic intent  
- classification- and policy-conditioned (pragmatics-aware)  
- avoids expensive LLM generation while still studying negotiation effects in a controlled way. ([ACL Anthology](https://aclanthology.org/2024.games-1.12.pdf?utm_source=chatgpt.com))

Classic foundations. We build on actor-critic RL (e.g., PPO/A2C) for stability and sample-efficiency, and on self-play/population training practices from multi-agent RL (e.g., league training / snapshotting) to manage non-stationarity.

**Proposed Approach**  
**Simulator** \- Use Catanatron for fast rollouts and reproducible seeds, wrapping it in a gym-like API with vectorized CPU workers. ([docs.catanatron.com](https://docs.catanatron.com/?utm_source=chatgpt.com))

**State** \- A compact, factorized observation with:  
(1) board tensors (hex resource types/numbers, ports, robber location)  
(2) graph/topology features for vertices/edges (buildings/roads)  
(3) per-player public features (VP, dev-card counts if public, longest road/largest army flags)  
(4) partially observed hand features (own exact hand; opponents: counts \+ belief features)  
(5) action history summary (recent trades/offers, robber events)

**Representation**

- Implement a cross-dimensional encoder  
- separate modules for faces/edges/vertices and per-player vectors  
- fused via attention-like pooling or gated aggregation (in the spirit of arXiv:2008.07079)  
- kept lightweight (tens of thousands to a few million parameters). ([arXiv](https://arxiv.org/abs/2008.07079?utm_source=chatgpt.com))

**Action space**

- Use hierarchical/parameterized actions to control branching macro-actions   
- E.g. (build road/settlement/city, buy dev card, move robber+steal target, propose/accept trade, end turn)   
- parameter heads choose locations/targets/resources.


**Algorithm**

- PPO with centralized critic \+ decentralized actors (CTDE) for multi-agent stability  
- actor conditioned on own observation  
- critic trained with richer features (including opponent public state) to reduce variance  
- Train via self-play with 4-player games and shared network weights across seats (symmetry \+ sample efficiency)

**Self-play schedule**

- Population-based training: maintain a pool of past snapshots \+ heuristic bots  
- sample opponents from a mixture to reduce overfitting/exploit-only strategies  
- Periodically freeze best checkpoints to serve as stable opponents and evaluation anchors.


**Curriculum** \- Stage training to improve sample efficiency:

1. opening curriculum focusing on initial placement/early game (short-horizon episodes),  
2. full-game training with shaped rewards,  
3. late-stage fine-tuning with reduced shaping \+ stronger opponent mix.

**Reward Design**

- Primary reward is win/placement-based (e.g., \+1 win, \+0.5 second, 0 otherwise) to align with Elo/win rate  
- Add mild potential-based shaping that is easy to ablate: Δ(VP), Δ(expected resource income from placements), penalties for illegal-action attempts, and small entropy regularization  
- explicitly test shaping removal to ensure the final policy does not depend on brittle reward hacks

**Opponent Modeling / Multi-Agent Considerations**

- Non-stationarity is addressed via:   
  - opponent pools and snapshot sampling  
  - seat randomization  
  - evaluation across unseen opponent mixtures and board seeds  
  - exploitability checks (train against A, evaluate against B/C)  
- optional opponent-embedding head trained to predict opponent archetypes (heuristic vs learned snapshot) from public behavior, used only as a conditioning signal (ablatable)

**Use of Search or Planning (if any)**  
To stay compute-light but improve long-horizon competence, add lightweight planning only for sparse, high-impact decisions:

- Initial placement planner: 1–2 step lookahead using a learned value proxy (critic) plus fast hand-crafted rollout heuristic for expected pips/port synergy/expansion options.  
- Critical build/robber turns: limited breadth lookahead over legal placements/targets (not full MCTS over whole turns).

This hybrid is chosen because full-tree search in Catan explodes (stochastic \+ negotiation), but targeted planning can yield large gains per compute and is easy to ablate for publishable insight.

**Optional negotiation module**

- Instead of LLM generation, implement a trade-offer policy conditioned on predicted dialogue acts / pragmatic intents (inspired by “LLMs of Catan”) and simple acceptance modeling  
- isolates the scientific question—*does pragmatics-aware trading improve win rate under self-play?*—without large-model cost. ([ACL Anthology](https://aclanthology.org/2024.games-1.12.pdf?utm_source=chatgpt.com))

**Evaluation Plan**  
**Baselines:** (i) strongest Catanatron heuristic bots available, (ii) random/legal bot, (iii) simple PPO baseline with flat encoding/no planning, (iv) best public agent available in the chosen framework when feasible (for Elo comparison). ([GitHub](https://github.com/bcollazo/catanatron?utm_source=chatgpt.com))

**Ablations:** cross-dimensional encoder → flat tensors; population self-play → naive self-play; planning on/off; shaping on/off; opponent embedding on/off; negotiation module on/off.

**Metrics** (win rate, Elo, sample efficiency, etc.)  
win rate and Elo vs strongest baselines over ≥5,000 evaluation games, reporting confidence intervals and significance tests (paired matchups). Secondary: games-to-target-Elo, trade efficiency (accepted beneficial trades, resource waste), robustness to dice variance and opponent adaptation, and stability (no collapse/exploit-only patterns)

**Offline vs online evaluation**

- Online: round-robin tournaments across opponent pools and unseen mixtures.   
- Offline: fixed seed suites for regression tests and debugging, plus “frozen-opponent” evaluations to detect non-stationary inflation. Generalization: held-out board seeds, held-out opponent mixtures, and cross-version evaluation against earlier checkpoints.

**Risks and Mitigations**  
Sparse credit assignment / instability?

- mitigate with curriculum \+ CTDE critic \+ conservative PPO clipping and KL control

Overfitting to opponent pool? 

- use diverse mixture (heuristics \+ snapshots), periodic introduction of new opponents, and held-out mixture tests

Negotiation exploits/collusion? 

- constrain trade space (rate limits, no side-channel), add fairness checks, and evaluate with/without negotiation enabled

Compute overruns?

- keep models small, prioritize CPU rollouts, and cap experiments to ≤10M steps with early stopping based on Elo plateaus

**Implementation Plan and Timeline**

Month 1: Integrate Catanatron; implement state/action masking; build heuristic \+ PPO baseline; reproduce stable training curves. ([docs.catanatron.com](https://docs.catanatron.com/?utm_source=chatgpt.com))  
Month 2: Add population self-play \+ snapshotting; implement cross-dimensional encoder; run sample-efficiency sweeps and representation ablations. ([arXiv](https://arxiv.org/abs/2008.07079?utm_source=chatgpt.com))  
Month 3: Add lightweight planning module; optional pragmatics-aware negotiation head; large-scale evaluation (≥5k games/matchup) \+ full ablation grid. ([ACL Anthology](https://aclanthology.org/2024.games-1.12.pdf?utm_source=chatgpt.com))  
Month 4: Paper draft \+ final experiments; robustness/generalization suite; open-source release with reproducible configs and tournament scripts.

Ethics & evaluation integrity \- We will explicitly test for deceptive, collusive, or exploitative negotiation (e.g., cycling trades to kingmake) by auditing trade logs and running “no-negotiation” evaluations as a control. Fairness is enforced via fixed opponent mixes, seat randomization, and public release of evaluation seeds to prevent benchmark leakage. Open-sourcing will include clear documentation, reproducible scripts, and a statement of intended use (research/benchmarking) while discouraging deployment in contexts that incentivize manipulative negotiation behavior.

**References (with links)**

1. Gendre, Q., & Kaneko, T. (2020). Playing Catan with Cross-dimensional Neural Network. arXiv:2008.07079. [https://arxiv.org/abs/2008.07079](https://arxiv.org/abs/2008.07079) ([arXiv](https://arxiv.org/abs/2008.07079?utm_source=chatgpt.com))  
2. Belle, N., Barnes, D., et al. (2025). Agents of Change: Self-Evolving LLM Agents for Strategic Planning. arXiv:2506.04651. [https://arxiv.org/abs/2506.04651](https://arxiv.org/abs/2506.04651) and project page [https://nbelle1.github.io/agents-of-change/](https://nbelle1.github.io/agents-of-change/) ([arXiv](https://arxiv.org/abs/2506.04651?utm_source=chatgpt.com))  
3. Martinenghi, A., et al. (2024). LLMs of Catan: Exploring Pragmatic Capabilities… (Games & NLP @ LREC-COLING). [https://aclanthology.org/2024.games-1.12.pdf](https://aclanthology.org/2024.games-1.12.pdf) ([ACL Anthology](https://aclanthology.org/2024.games-1.12.pdf?utm_source=chatgpt.com))  
4. Catanatron documentation and repository. [https://docs.catanatron.com/](https://docs.catanatron.com/) and [https://github.com/bcollazo/catanatron](https://github.com/bcollazo/catanatron) ([docs.catanatron.com](https://docs.catanatron.com/?utm_source=chatgpt.com))

