# Week 5 Multi-Agent Diagnostics Add-On

This add-on supplies the missing random-policy baseline and representative
trajectory inspection for the Simple Spread benchmark.

## Installation

Place the script at:

```text
W05_RL_Benchmarks/scripts/evaluate_multi_agent_diagnostics.py
```

It imports the existing `common.py` and `train_multi_agent.py`, so do not place
it outside the existing `scripts/` folder.

## Run

From the `W05_RL_Benchmarks` root, with `.venv_w05` activated:

```bash
python scripts/evaluate_multi_agent_diagnostics.py \
  --profile standard
```

The script uses:

- training seeds `7, 17, 27, 37, 47`
- 10 held-out episodes per policy and training seed
- identical episode seeds for random and trained PPO
- deterministic PPO actions
- seed-level means as the main replication units

It does **not** retrain any PPO model.

## Outputs

```text
results/
  multi_agent_diagnostics_episode_results.csv
  multi_agent_diagnostics_seed_summary.csv
  multi_agent_diagnostics_overall_summary.csv
  multi_agent_diagnostics_paired_deltas.csv
  multi_agent_diagnostics_metadata.json
  multi_agent_diagnostics_summary.md
  multi_agent_trajectory_summary.json
  simple_spread_trajectory_seed<seed>_episode0_random.csv
  simple_spread_trajectory_seed<seed>_episode0_ppo.csv

plots/
  simple_spread_random_vs_ppo_return.png
  simple_spread_random_vs_ppo_assignment_distance.png
  simple_spread_random_vs_ppo_collisions.png
  simple_spread_trajectory_seed<seed>_episode0_random.png
  simple_spread_trajectory_seed<seed>_episode0_ppo.png
```

## Interpretation

- Higher return favors PPO.
- Lower assignment distance favors PPO.
- Lower collision rate favors PPO.
- The trajectory seed is selected as the median seed by the original standard
  benchmark's final assignment distance, unless manually overridden.
- The 0.25 coordination threshold is an analyst-defined diagnostic, not an
  official MPE2 success criterion.
- A random baseline can show whether PPO improved over untrained actions, but
  it does not prove robust role assignment or transfer to physical robots.
