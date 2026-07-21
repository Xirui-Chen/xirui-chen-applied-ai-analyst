# Multi-Agent Random Baseline and Trajectory Diagnostic

## Design

The diagnostic re-evaluated each of the five trained parameter-sharing PPO
policies against a random-action policy. PPO and random actions were tested on
the same 10 held-out episode seeds within each training seed. The five
training-seed means, rather than all 50 episodes, were treated as the main
replication units.

## Main Results

| Metric | Random policy, mean ± SD across seeds | Trained PPO, mean ± SD across seeds |
|---|---:|---:|
| Mean agent return | -23.727 ± 3.304 | -21.158 ± 4.803 |
| Final assignment distance | 0.919 ± 0.081 | 0.682 ± 0.221 |
| Collision pairs per cycle | 0.046 ± 0.019 | 0.074 ± 0.036 |

PPO-minus-random differences were also saved per training seed in
`multi_agent_diagnostics_paired_deltas.csv`. Positive return differences favor
PPO. Negative assignment-distance and collision differences favor PPO.

## Trajectory Inspection

The representative trajectory used training seed 27, chosen
as the median seed by the original standard-run final assignment distance. The
matched random and PPO trajectories use the same held-out episode seed
(902700), reducing the risk that the visual comparison is
driven only by different initial conditions.

## Interpretation Boundary

This diagnostic tests whether the trained parameter-sharing PPO baseline
outperformed random actions under the same Simple Spread initial conditions. It
does not establish robust multi-robot coordination, role specialization, or
transfer to fielded robots. The 0.25 coordination threshold remains an
analyst-defined diagnostic rather than an official environment success
criterion.
