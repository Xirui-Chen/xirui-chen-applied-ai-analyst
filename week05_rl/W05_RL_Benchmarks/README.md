# W05 RL Benchmarks

## Purpose

This folder contains the Week 5 reinforcement-learning experiment pipeline for the Applied AI Analyst internship. It runs:

- PPO and SAC on `LunarLanderContinuous-v3` and `BipedalWalker-v3`
- five seeds for every environment-algorithm combination
- a parameter-sharing PPO benchmark on Simple Spread with five seeds
- deterministic evaluation curves, wall-clock measurements, sample-efficiency summaries, seeded result CSVs, TensorBoard logs, and plots

The work is an **analyst benchmark**, not a robotics research contribution. The environments are small simulated control tasks. Results can support disciplined discussion of optimization stability, sample efficiency, and coordination, but they do not establish performance on physical robots.

## Experiment Grid

### Single-agent benchmark

| Environment | Algorithms | Seeds |
|---|---|---:|
| LunarLander continuous control | PPO, SAC | 7, 17, 27, 37, 47 |
| BipedalWalker-v3 | PPO, SAC | 7, 17, 27, 37, 47 |

Total: `2 environments × 2 algorithms × 5 seeds = 20 runs`.

The work plan names `LunarLanderContinuous-v3`. Current Gymnasium documentation exposes continuous control through `LunarLander-v3` with `continuous=True`. The training helper first tries the requested ID, then records and uses the documented fallback when necessary.

### Multi-agent benchmark

Simple Spread uses three homogeneous agents and three landmarks. One shared PPO policy controls all agents through a SuperSuit vector-environment adapter. The benchmark records:

- mean agent return
- one-agent-per-landmark assignment distance
- coordination success rate under an explicitly analyst-defined distance threshold
- collision-pair rate

This is parameter sharing with a single-agent PPO implementation. It is not independent PPO, centralized-critic PPO, MAPPO, or another general multi-agent algorithm.

## Folder Structure

```text
W05_RL_Benchmarks/
├── README.md
├── requirements_w05.txt
├── configs/
│   └── w05_rl_config.yaml
├── scripts/
│   ├── common.py
│   ├── check_environment.py
│   ├── train_single_agent.py
│   ├── train_multi_agent.py
│   ├── aggregate_results.py
│   ├── plot_learning_curves.py
│   └── run_all.py
├── results/
├── tensorboard/
├── plots/
└── models/
```

Result CSVs, plots, and TensorBoard event files appear after training.

## 1. Copy the Folder into the Repository

Recommended location:

```text
xirui-chen-applied-ai-analyst/
└── week05_rl/
    └── W05_RL_Benchmarks/
```

Then enter the benchmark folder:

```bash
cd ~/Downloads/xirui-chen-applied-ai-analyst/week05_rl/W05_RL_Benchmarks
```

## 2. Create a Clean Python 3.11 Environment

```bash
python3.11 -m venv .venv_w05
source .venv_w05/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_w05.txt
```

On macOS, if Box2D still fails to build:

```bash
brew install swig
python -m pip install --no-cache-dir "gymnasium[box2d]"
```

## 3. Verify the Environment

```bash
python scripts/check_environment.py
```

The check should successfully create continuous Lunar Lander, Bipedal Walker, Simple Spread, and the SuperSuit-to-SB3 vector adapter. Do not start the full grid until this passes.

## 4. Run a Smoke Test

A smoke test checks plumbing, not learning quality:

```bash
python scripts/run_all.py \
  --profile smoke \
  --seeds 7 \
  --include single multi \
  --progress
```

After it finishes:

```bash
find results -type f | sort
find plots -type f | sort
```

## 5. Run the Required Five-Seed Benchmark

```bash
python scripts/run_all.py \
  --profile standard \
  --include single multi \
  --progress
```

This runs all 25 jobs sequentially: 20 single-agent runs and 5 Simple Spread runs. Completed runs are skipped automatically because each training script checks for `run_summary.json`. If a run is interrupted, rerun the same command. Use `--overwrite` only when intentionally replacing completed results.

To continue past one failed run:

```bash
python scripts/run_all.py \
  --profile standard \
  --include single multi \
  --progress \
  --continue-on-error
```

## 6. Run Individual Jobs

Single-agent example:

```bash
python scripts/train_single_agent.py \
  --environment LunarLanderContinuous-v3 \
  --algorithm PPO \
  --seed 7 \
  --profile standard \
  --progress
```

Multi-agent example:

```bash
python scripts/train_multi_agent.py \
  --seed 7 \
  --profile standard \
  --progress
```

## 7. Rebuild Tables and Figures

These commands do not retrain models:

```bash
python scripts/aggregate_results.py
python scripts/plot_learning_curves.py
```

## 8. Open TensorBoard

```bash
tensorboard --logdir tensorboard
```

Open the local URL printed in Terminal, usually `http://localhost:6006`.

## Metrics

### Convergence stability

The aggregation script reports final deterministic-evaluation return across seeds and its standard deviation. It also reports `std / abs(mean)` as a compact stability diagnostic. Because returns may be negative, interpret it only alongside the raw mean and standard deviation.

### Wall-clock cost

Every run records total training seconds, environment steps per second, and wall-clock time at each evaluation checkpoint. Wall-clock comparisons are machine-specific and are valid only when runs use the same machine and comparable background load.

### Sample efficiency

The pipeline records reward at 25%, 50%, and 75% of the training budget, normalized area under the evaluation learning curve, and the first timestep reaching the environment's documented solution threshold when reached. The configured thresholds are 200 for Lunar Lander and 300 for normal Bipedal Walker.

### Coordination behavior

For Simple Spread, the benchmark uses a minimum-cost assignment between agent positions and landmark positions as an interpretable coverage proxy. It also records collision pairs. The `0.25` success-distance threshold is an analyst-defined diagnostic, not an official PettingZoo success criterion.

## Reproducibility Notes

- The five seeds are fixed in `configs/w05_rl_config.yaml`.
- Training and evaluation use different seed ranges.
- Evaluations use deterministic policy actions.
- Every run stores configuration, software versions, machine metadata, and output paths.
- No claim should be based on the smoke profile.
- The standard profile is still a limited training budget. Failure to converge may reflect budget or hyperparameter limitations, not an inherent algorithm failure.

## What to Commit

Commit scripts, YAML configuration, README, seeded result CSVs, learning-curve PNGs, and TensorBoard logs if repository size remains reasonable. Usually do not commit `.venv_w05/`, saved model weights, or Python cache files.

## AI Assistance Disclosure

AI assistance was used to draft portions of the experiment scaffolding, documentation, and plotting utilities. The intern is responsible for running, validating, interpreting, and revising all experiments and claims.
