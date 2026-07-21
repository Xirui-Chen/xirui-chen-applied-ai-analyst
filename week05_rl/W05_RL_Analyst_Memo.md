# RL Analyst Memo

**Subject:** Transferable RL Lessons for HTD-IRL and CRL-MRS  
**Scope:** Week 5 reinforcement-learning benchmarks  

## Executive Summary

This memo summarizes a controlled reinforcement-learning benchmark intended to build analyst-level understanding of the concepts behind **HTD-IRL**, framed here as hierarchical task decomposition and inverse reinforcement learning, and **CRL-MRS**, framed here as continual reinforcement learning and multi-robot scaling. It does not evaluate a proprietary InGen implementation and should not be treated as a research contribution or evidence of field-robot performance.

I ran PPO and SAC on `LunarLanderContinuous-v3` and `BipedalWalker-v3` across five fixed seeds, then ran a parameter-sharing PPO baseline on PettingZoo/MPE2 `simple_spread_v3`. SAC achieved stronger final returns and more benchmark-threshold crossings, especially on BipedalWalker, but required roughly 12 to 14 times more wall-clock time than PPO on the same CPU machine. Its stability also depended on the task: SAC was highly consistent on BipedalWalker but had one failed LunarLander seed. PPO trained much faster, but its BipedalWalker runs showed large cross-seed variation and one case of late-stage policy regression.

The multi-agent result was more limited. Relative to matched random-action baselines, PPO improved one-to-one landmark assignment distance in all five seeds and improved mean return in four of five. However, neither PPO nor random policies reached the analyst-defined coordination threshold, and PPO increased average collision frequency. A matched trajectory showed more purposeful motion under PPO, but not stable one-agent-per-landmark role allocation.

Three lessons follow:

1. **Choose the learning regime by hierarchy level and operating constraint, not by a single benchmark winner.**
2. **Evaluate behavior and policy stability, not only aggregate reward.**
3. **Parameter sharing is a useful baseline, but scalable multi-robot coordination requires explicit credit, role, and continual-adaptation structure.**

---

## 1. Benchmark Design and Evidence Boundary

### Single-agent benchmark

| Environment | Algorithms | Seeds | Training budget |
|---|---|---:|---:|
| LunarLander continuous control | PPO, SAC | 7, 17, 27, 37, 47 | 150,000 timesteps |
| BipedalWalker-v3 | PPO, SAC | 7, 17, 27, 37, 47 | 300,000 timesteps |

Each run used deterministic evaluation on a separately seeded environment. I recorded final and best return, learning curves, threshold attainment, sample-efficiency checkpoints, wall-clock time, and throughput. All timing results were measured on one Apple Silicon `arm64` machine using CPU execution.

PPO is an on-policy policy-gradient method built around a clipped surrogate objective [1]. SAC is an off-policy actor-critic method that uses experience replay and a maximum-entropy objective [2]. The benchmark used one fixed configuration per algorithm and did not perform broad hyperparameter tuning. The results therefore describe these reproducible baselines, not universal algorithm rankings.

### Multi-agent benchmark

The Simple Spread experiment used three homogeneous agents, three landmarks, and one shared PPO policy. This is a **parameter-sharing PPO baseline**, not MAPPO, MADDPG, COMA, or QMIX.

Five trained PPO seeds were compared with random actions on ten matched held-out episodes per seed. Diagnostics included:

- mean agent return;
- one-to-one agent-landmark assignment distance;
- final coordination success at distance `<= 0.25`;
- collision pairs per cycle;
- duplicate nearest-landmark behavior; and
- nearest-landmark switching.

The `0.25` threshold is an analyst-defined diagnostic, not an official MPE2 solution criterion.

### What the evidence supports

The experiment demonstrates task- and seed-dependent differences in return, stability, compute cost, and coordination behavior. It does **not** establish performance on physical robots, prove hierarchical decomposition, recover a reward through IRL, test continual learning, or validate scalable multi-robot deployment.

---

## 2. Quantitative Results

### 2.1 PPO versus SAC

**Table 1. Five-seed results, mean ± SD across seeds**

| Environment | Algorithm | Final return | Threshold reached | Median steps to threshold* | Mean wall-clock |
|---|---|---:|---:|---:|---:|
| BipedalWalker-v3 | PPO | 180.57 ± 135.76 | 0/5 | Not reached | 91.0 s |
| BipedalWalker-v3 | SAC | **306.96 ± 5.50** | **4/5** | 230,000 | 1,063.3 s |
| LunarLander continuous | PPO | 80.46 ± 39.55 | 0/5 | Not reached | 36.2 s |
| LunarLander continuous | SAC | **159.75 ± 137.17** | **3/5** | 70,000 | 490.9 s |

\*Among successful runs. Thresholds were 300 for BipedalWalker and 200 for LunarLander.

On BipedalWalker, SAC produced a 70% higher mean final return than PPO and dramatically lower cross-seed dispersion. It also required 11.7 times more wall-clock time. PPO was cheaper but unstable. Seed 47 reached a best return of 260.06 and later ended at -54.87, showing that an apparently strong checkpoint did not guarantee a stable final policy.

On LunarLander, SAC nearly doubled PPO's mean final return but had much greater dispersion. Three SAC seeds crossed the 200 threshold, while seed 27 ended at -73.70. PPO did not solve the environment within the selected budget, but its final outcomes were less dispersed than SAC's.

Relevant figures:

- `plots/BipedalWalker-v3_standard_ppo_sac_learning_curves.png`
- `plots/LunarLanderContinuous-v3_standard_ppo_sac_learning_curves.png`
- `plots/single_agent_standard_wallclock_tradeoff.png`

### 2.2 Simple Spread coordination

The original PPO benchmark ended at a mean agent return of **-22.22 ± 1.85**, final assignment distance of **0.667 ± 0.109**, and coordination success of **0/5 seeds**.

**Table 2. Matched random-policy diagnostic, mean ± SD across five seed means**

| Metric | Random policy | Trained PPO | Result |
|---|---:|---:|---|
| Mean agent return | -23.73 ± 3.30 | **-21.16 ± 4.80** | PPO higher |
| Final assignment distance | 0.919 ± 0.081 | **0.682 ± 0.221** | PPO lower |
| Collision pairs per cycle | **0.046 ± 0.019** | 0.074 ± 0.036 | PPO higher |
| Coordination success rate | 0% | 0% | No difference |

PPO improved return in four of five seeds and reduced assignment distance in all five. With only five paired seed-level observations, two-sided Wilcoxon tests did not support significance claims: return `p = 0.125`, assignment distance `p = 0.0625`, and collision rate `p = 0.4375`. The assignment improvement was directionally consistent, but not statistically established.

In the matched seed-27 trajectory, PPO improved return from -31.74 to -30.53 and reduced final assignment distance from 0.968 to 0.839, while collisions increased from 0.00 to 0.16 per cycle. The trajectory shows more directed movement but incomplete landmark coverage.

Relevant figures:

- `plots/simple_spread_random_vs_ppo_return.png`
- `plots/simple_spread_random_vs_ppo_assignment_distance.png`
- `plots/simple_spread_random_vs_ppo_collisions.png`
- `plots/simple_spread_trajectory_seed27_episode0_random.png`
- `plots/simple_spread_trajectory_seed27_episode0_ppo.png`

---

## 3. Three Transferable Lessons

## Lesson 1: Match the learning regime to the hierarchy level and operating constraint

### Experimental evidence

No algorithm dominated every relevant dimension. SAC was stronger on final BipedalWalker performance and threshold attainment but was roughly 12 times slower. On LunarLander, it offered higher upside with greater seed sensitivity. PPO was computationally lighter but could be unstable and regress late in training.

The practical conclusion is not that SAC is always better. It is that data reuse, update cadence, compute budget, and acceptable variance should be chosen for the decision layer being trained.

### Literature connection

The options framework formalizes temporal abstraction through policies that act over extended time periods rather than only primitive actions [3]. HIRO separates a higher-level goal policy from a lower-level controller and shows why off-policy hierarchical training needs correction when changes in the lower-level policy alter the effective action space seen by the higher level [4].

### Implication for HTD-IRL and CRL-MRS

A conceptual HTD-IRL design should define separate contracts for each level:

- **High level:** slower updates, longer evaluation horizons, subgoal success, plan consistency, and decomposition quality.
- **Low level:** faster control, local robustness, sample reuse, and safety-constraint compliance.
- **Interface:** versioned subgoal semantics and correction or relabeling when the lower-level controller changes.

For CRL-MRS, offline fleet training may tolerate heavier replay and longer optimization, while edge adaptation may require bounded compute and conservative updates. Model selection should therefore use a Pareto view of return, stability, wall-clock cost, and data consumption.

---

## Lesson 2: Evaluate behavior and policy stability, not only aggregate reward

### Experimental evidence

Two forms of metric mismatch appeared. First, BipedalWalker PPO seed 47 had a strong intermediate checkpoint but a negative final result. Second, Simple Spread PPO improved reward and assignment distance while increasing collisions and failing the coordination threshold.

A higher aggregate score therefore did not guarantee a stable, complete, or safer behavior.

### Literature connection

Inverse reinforcement learning infers a reward that explains expert behavior rather than relying entirely on a hand-written scalar objective [5]. However, multiple rewards may be compatible with the same demonstrations, and reward transfer depends on assumptions about the expert and target environment [6]. Continual RL introduces a related evaluation problem: an agent must adapt under non-stationarity while retaining prior capabilities, so a single final score cannot capture the full lifecycle [7].

The present benchmark did not perform IRL or continual learning. It does show why those systems need broader validation than a reward curve alone.

### Implication for HTD-IRL and CRL-MRS

HTD-IRL should treat inferred reward and subgoal decomposition as hypotheses to test. Evaluation should include:

- task completion;
- safety violations and collisions;
- energy and time cost;
- intervention frequency;
- subgoal churn;
- low-level feasibility; and
- robustness under perturbation.

CRL-MRS should additionally track retention, worst-scenario performance, checkpoint regression, policy rollback, and compatibility across robot versions. A policy update that improves mean reward but worsens collision risk should not automatically replace the incumbent.

---

## Lesson 3: Shared policies can improve average behavior, but coordination needs explicit structure

### Experimental evidence

Parameter-sharing PPO learned behavior that was better than random action on return and assignment distance. The assignment-distance improvement appeared in all five seeds. It still did not produce reliable role allocation, and its improved movement came with a higher average collision rate.

The supported conclusion is narrow: parameter sharing is a credible baseline for homogeneous agents, but it is not evidence of robust multi-robot coordination.

### Literature connection

Multi-agent learning is difficult because other learning agents make the environment non-stationary from an individual policy's perspective [8]. Centralized-training and decentralized-execution methods use joint information during training while preserving local execution. COMA uses counterfactual baselines for agent-level credit assignment [9]. QMIX factorizes a centralized joint value function into decentralized values [10]. MAPPO results show that PPO-based methods can be strong cooperative baselines, but also stress implementation and training choices rather than suggesting that ordinary parameter sharing solves the full problem [11].

### Implication for CRL-MRS

A scalable CRL-MRS design should explicitly provide:

1. **Agent context:** identity, capability, health, location, or assigned role when robots are not interchangeable.
2. **Credit assignment:** evidence linking team outcomes to individual actions.
3. **Training architecture:** centralized critics, value factorization, or another CTDE mechanism tested against shared PPO.
4. **Continual adaptation:** support for changing team composition, new tasks, communication loss, and retention of previously reliable behavior.

HTD-IRL needs similar structure at the task level. Stable specialization should come from explicit subgoals and handoff conditions, not from the assumption that a shared reward will automatically create roles.

---

## 4. Recommendation and Next Step

The Week 5 benchmark supports further testing, not deployment. The next experimental step should compare parameter-sharing PPO with at least one true MARL baseline, such as IPPO/MAPPO or a centralized-critic method, using the same seeds and matched episode sets. A future hierarchy experiment should compare flat control with a high-level subgoal policy and report both task return and behavior-level safety metrics.

The most important conclusion is that HTD-IRL and CRL-MRS should not be designed around a single optimizer or leaderboard metric. They need level-specific learning regimes, behavior-aware evaluation, explicit coordination structure, and lifecycle controls for policy regression and continual adaptation.

These are transferable design requirements derived from a literature-anchored analyst exercise. They are not claims that toy-environment performance will transfer directly to fielded robots.

---

## References

[1] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347. https://arxiv.org/abs/1707.06347

[2] Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. Proceedings of the 35th International Conference on Machine Learning, PMLR 80, 1861-1870. https://proceedings.mlr.press/v80/haarnoja18b.html

[3] Sutton, R. S., Precup, D., & Singh, S. (1999). *Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning*. Artificial Intelligence, 112(1-2), 181-211. https://doi.org/10.1016/S0004-3702(99)00052-1

[4] Nachum, O., Gu, S., Lee, H., & Levine, S. (2018). *Data-Efficient Hierarchical Reinforcement Learning*. Advances in Neural Information Processing Systems 31. https://proceedings.neurips.cc/paper/2018/hash/e6384711491713d29bc63fc5eeb5ba4f-Abstract.html

[5] Ng, A. Y., & Russell, S. J. (2000). *Algorithms for Inverse Reinforcement Learning*. Proceedings of the 17th International Conference on Machine Learning, 663-670. https://ai.stanford.edu/~ang/papers/icml00-irl.pdf

[6] Metelli, A. M., Ramponi, G., Concetti, A., & Restelli, M. (2021). *Provably Efficient Learning of Transferable Rewards*. Proceedings of the 38th International Conference on Machine Learning, PMLR 139, 7665-7676. https://proceedings.mlr.press/v139/metelli21a.html

[7] Khetarpal, K., Riemer, M., Rish, I., & Precup, D. (2020). *Towards Continual Reinforcement Learning: A Review and Perspectives*. arXiv:2012.13490. https://arxiv.org/abs/2012.13490

[8] Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I. (2017). *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments*. Advances in Neural Information Processing Systems 30. https://proceedings.neurips.cc/paper/2017/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html

[9] Foerster, J., Farquhar, G., Afouras, T., Nardelli, N., & Whiteson, S. (2018). *Counterfactual Multi-Agent Policy Gradients*. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1). https://arxiv.org/abs/1705.08926

[10] Rashid, T., Samvelyan, M., Schroeder, C., Farquhar, G., Foerster, J., & Whiteson, S. (2018). *QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning*. Proceedings of the 35th International Conference on Machine Learning, PMLR 80, 4295-4304. https://proceedings.mlr.press/v80/rashid18a.html

[11] Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y., Bayen, A., & Wu, Y. (2022). *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*. Advances in Neural Information Processing Systems 35. https://papers.nips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html
