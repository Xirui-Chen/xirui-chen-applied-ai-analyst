#!/usr/bin/env python3
"""
Evaluate trained Simple Spread PPO policies against a reproducible random-policy
baseline and create trajectory diagnostics.

This script does not retrain any model. It:
1. Loads the five standard-profile PPO models.
2. Evaluates PPO and random actions on exactly the same episode seeds.
3. Reports return, assignment distance, coordination success, and collisions.
4. Produces paired comparison plots.
5. Produces matched random/PPO trajectory plots for one representative seed.

Run from the W05_RL_Benchmarks root:
    python scripts/evaluate_multi_agent_diagnostics.py --profile standard
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from stable_baselines3 import PPO

from common import DEFAULT_CONFIG, PROJECT_ROOT, load_config, system_metadata, write_json
from train_multi_agent import make_parallel_env


PolicyKind = Literal["random", "ppo"]


def extract_geometry(
    observations: dict[str, np.ndarray],
    n_agents: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Recover agent and landmark coordinates from Simple Spread observations.

    MPE2 documents the observation order as:
    [self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, communication].
    """
    agents = sorted(observations)
    if not agents:
        raise ValueError("No active agents were present in the observation dictionary.")

    agent_positions = np.vstack(
        [np.asarray(observations[agent], dtype=float)[2:4] for agent in agents]
    )

    first_obs = np.asarray(observations[agents[0]], dtype=float)
    first_agent_position = first_obs[2:4]
    landmark_relative = first_obs[4 : 4 + 2 * n_agents].reshape(n_agents, 2)
    landmark_positions = first_agent_position + landmark_relative

    return agents, agent_positions, landmark_positions


def state_diagnostics(
    observations: dict[str, np.ndarray],
    n_agents: int,
) -> dict[str, Any]:
    """Compute assignment, nearest-landmark, and collision diagnostics."""
    agents, agent_positions, landmark_positions = extract_geometry(
        observations,
        n_agents=n_agents,
    )

    distances = np.linalg.norm(
        agent_positions[:, None, :] - landmark_positions[None, :, :],
        axis=2,
    )

    assignment_rows, assignment_cols = linear_sum_assignment(distances)
    assignment_map = {
        agents[int(row)]: int(col)
        for row, col in zip(assignment_rows, assignment_cols)
    }
    assignment_distance = float(
        np.mean(distances[assignment_rows, assignment_cols])
    )

    nearest_landmark = {
        agents[index]: int(np.argmin(distances[index]))
        for index in range(len(agents))
    }
    duplicate_nearest_count = len(agents) - len(set(nearest_landmark.values()))

    collision_pairs = 0
    for first in range(len(agent_positions)):
        for second in range(first + 1, len(agent_positions)):
            if np.linalg.norm(
                agent_positions[first] - agent_positions[second]
            ) < 0.30:
                collision_pairs += 1

    return {
        "agents": agents,
        "agent_positions": agent_positions,
        "landmark_positions": landmark_positions,
        "assignment_map": assignment_map,
        "mean_assignment_distance": assignment_distance,
        "nearest_landmark": nearest_landmark,
        "duplicate_nearest_count": duplicate_nearest_count,
        "collision_pairs": collision_pairs,
    }


def seed_action_spaces(env, episode_seed: int) -> None:
    """Seed every random action space so the random baseline is reproducible."""
    for index, agent in enumerate(env.possible_agents):
        env.action_space(agent).seed(episode_seed + 10_000 + index)


def rollout_episode(
    policy_kind: PolicyKind,
    env_cfg: dict[str, Any],
    episode_seed: int,
    model: PPO | None = None,
    record_trajectory: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run one matched Simple Spread evaluation episode."""
    if policy_kind == "ppo" and model is None:
        raise ValueError("A trained PPO model is required for policy_kind='ppo'.")

    env, environment_source = make_parallel_env(env_cfg, seed=episode_seed)
    observations, _ = env.reset(seed=episode_seed)
    seed_action_spaces(env, episode_seed)

    n_agents = int(env_cfg["n_agents"])
    success_distance = float(env_cfg["coordination_success_distance"])
    agent_returns = {agent: 0.0 for agent in env.possible_agents}

    assignment_distances: list[float] = []
    collision_pairs_total = 0
    duplicate_nearest_total = 0
    nearest_switches_total = 0
    nearest_switch_opportunities = 0
    previous_nearest: dict[str, int] | None = None
    trajectory_rows: list[dict[str, Any]] = []
    cycles = 0

    def capture_state(
        step: int,
        current_observations: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        diagnostics = state_diagnostics(current_observations, n_agents=n_agents)
        if record_trajectory:
            for agent_index, agent in enumerate(diagnostics["agents"]):
                position = diagnostics["agent_positions"][agent_index]
                trajectory_rows.append(
                    {
                        "policy": policy_kind,
                        "episode_seed": episode_seed,
                        "step": step,
                        "entity_type": "agent",
                        "entity_id": agent,
                        "x": float(position[0]),
                        "y": float(position[1]),
                        "assigned_landmark": diagnostics["assignment_map"][agent],
                        "nearest_landmark": diagnostics["nearest_landmark"][agent],
                    }
                )

            for landmark_index, position in enumerate(
                diagnostics["landmark_positions"]
            ):
                trajectory_rows.append(
                    {
                        "policy": policy_kind,
                        "episode_seed": episode_seed,
                        "step": step,
                        "entity_type": "landmark",
                        "entity_id": f"landmark_{landmark_index}",
                        "x": float(position[0]),
                        "y": float(position[1]),
                        "assigned_landmark": landmark_index,
                        "nearest_landmark": landmark_index,
                    }
                )
        return diagnostics

    current_diagnostics = capture_state(0, observations)
    previous_nearest = current_diagnostics["nearest_landmark"]

    while env.agents:
        actions = {}
        for agent in env.agents:
            if policy_kind == "random":
                actions[agent] = env.action_space(agent).sample()
            else:
                action, _ = model.predict(
                    observations[agent],
                    deterministic=True,
                )
                actions[agent] = action

        (
            next_observations,
            rewards,
            terminations,
            truncations,
            _,
        ) = env.step(actions)

        for agent, reward in rewards.items():
            agent_returns[agent] += float(reward)

        cycles += 1
        if next_observations:
            current_diagnostics = capture_state(cycles, next_observations)
            assignment_distances.append(
                current_diagnostics["mean_assignment_distance"]
            )
            collision_pairs_total += current_diagnostics["collision_pairs"]
            duplicate_nearest_total += current_diagnostics[
                "duplicate_nearest_count"
            ]

            if previous_nearest is not None:
                common_agents = set(previous_nearest).intersection(
                    current_diagnostics["nearest_landmark"]
                )
                nearest_switches_total += sum(
                    previous_nearest[agent]
                    != current_diagnostics["nearest_landmark"][agent]
                    for agent in common_agents
                )
                nearest_switch_opportunities += len(common_agents)

            previous_nearest = current_diagnostics["nearest_landmark"]

        observations = next_observations

        if all(
            terminations.get(agent, False) or truncations.get(agent, False)
            for agent in set(terminations) | set(truncations)
        ):
            break

    env.close()

    final_assignment_distance = (
        float(assignment_distances[-1])
        if assignment_distances
        else float("nan")
    )

    result = {
        "policy": policy_kind,
        "episode_seed": episode_seed,
        "mean_agent_return": float(np.mean(list(agent_returns.values()))),
        "team_return_sum": float(np.sum(list(agent_returns.values()))),
        "mean_assignment_distance": float(np.mean(assignment_distances))
        if assignment_distances
        else float("nan"),
        "final_assignment_distance": final_assignment_distance,
        "coordination_success": int(
            np.isfinite(final_assignment_distance)
            and final_assignment_distance <= success_distance
        ),
        "collision_pairs_per_cycle": float(
            collision_pairs_total / max(cycles, 1)
        ),
        "duplicate_nearest_landmarks_per_cycle": float(
            duplicate_nearest_total / max(cycles, 1)
        ),
        "nearest_landmark_switch_rate": float(
            nearest_switches_total / max(nearest_switch_opportunities, 1)
        ),
        "cycles": cycles,
        "environment_source": environment_source,
    }

    return result, pd.DataFrame(trajectory_rows)


def model_path_for_seed(seed: int, profile: str) -> Path:
    """Return the expected trained PPO model path."""
    return (
        PROJECT_ROOT
        / "models"
        / "multi_agent"
        / f"simple_spread_v3__PPO__seed{seed}__{profile}"
        / "model.zip"
    )


def paired_episode_seed(training_seed: int, episode_index: int) -> int:
    """Create a deterministic held-out episode seed shared by PPO and random."""
    return 900_000 + training_seed * 100 + episode_index


def evaluate_all(
    config: dict[str, Any],
    profile: str,
    seeds: list[int],
    n_episodes: int,
    device: str,
) -> pd.DataFrame:
    """Evaluate all trained seeds against matched random-policy episodes."""
    env_cfg = config["multi_agent"]
    rows: list[dict[str, Any]] = []

    for training_seed in seeds:
        model_path = model_path_for_seed(training_seed, profile)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing trained model for seed {training_seed}: {model_path}"
            )

        print(f"Loading PPO model for training seed {training_seed}: {model_path}")
        model = PPO.load(str(model_path), device=device)

        for episode_index in range(n_episodes):
            episode_seed = paired_episode_seed(
                training_seed,
                episode_index,
            )

            for policy_kind in ("random", "ppo"):
                result, _ = rollout_episode(
                    policy_kind=policy_kind,
                    env_cfg=env_cfg,
                    episode_seed=episode_seed,
                    model=model if policy_kind == "ppo" else None,
                    record_trajectory=False,
                )
                result["training_seed"] = training_seed
                result["episode_index"] = episode_index
                result["profile"] = profile
                rows.append(result)

        print(
            f"Completed matched random/PPO evaluation for seed {training_seed} "
            f"({n_episodes} episodes per policy)."
        )

    return pd.DataFrame(rows)


def create_seed_summary(episode_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize episodes within each training seed and policy."""
    metric_columns = [
        "mean_agent_return",
        "team_return_sum",
        "mean_assignment_distance",
        "final_assignment_distance",
        "coordination_success",
        "collision_pairs_per_cycle",
        "duplicate_nearest_landmarks_per_cycle",
        "nearest_landmark_switch_rate",
    ]

    summary = (
        episode_df.groupby(
            ["profile", "training_seed", "policy"],
            as_index=False,
        )[metric_columns]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "_".join(part for part in column if part)
        for column in summary.columns.to_flat_index()
    ]
    return summary


def create_overall_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize across the five training seeds.

    Seed-level means are used as the unit of replication, rather than treating
    all episodes as independent training runs.
    """
    mean_columns = [
        column
        for column in seed_summary.columns
        if column.endswith("_mean")
        and column not in {"training_seed_mean"}
    ]

    records: list[dict[str, Any]] = []
    for (profile, policy), group in seed_summary.groupby(
        ["profile", "policy"],
    ):
        record: dict[str, Any] = {
            "profile": profile,
            "policy": policy,
            "n_training_seeds": int(group["training_seed"].nunique()),
        }
        for column in mean_columns:
            record[f"{column}_across_seed_mean"] = float(group[column].mean())
            record[f"{column}_across_seed_std"] = float(
                group[column].std(ddof=1)
            )
        records.append(record)

    return pd.DataFrame(records)


def create_paired_seed_deltas(seed_summary: pd.DataFrame) -> pd.DataFrame:
    """Compute PPO minus random differences for each training seed."""
    indexed = seed_summary.set_index(
        ["profile", "training_seed", "policy"]
    )
    rows = []

    for profile, training_seed in sorted(
        {
            (index[0], index[1])
            for index in indexed.index
        }
    ):
        random_row = indexed.loc[(profile, training_seed, "random")]
        ppo_row = indexed.loc[(profile, training_seed, "ppo")]
        rows.append(
            {
                "profile": profile,
                "training_seed": training_seed,
                "delta_mean_agent_return_ppo_minus_random": float(
                    ppo_row["mean_agent_return_mean"]
                    - random_row["mean_agent_return_mean"]
                ),
                "delta_final_assignment_distance_ppo_minus_random": float(
                    ppo_row["final_assignment_distance_mean"]
                    - random_row["final_assignment_distance_mean"]
                ),
                "delta_collision_pairs_per_cycle_ppo_minus_random": float(
                    ppo_row["collision_pairs_per_cycle_mean"]
                    - random_row["collision_pairs_per_cycle_mean"]
                ),
                "delta_duplicate_nearest_landmarks_ppo_minus_random": float(
                    ppo_row[
                        "duplicate_nearest_landmarks_per_cycle_mean"
                    ]
                    - random_row[
                        "duplicate_nearest_landmarks_per_cycle_mean"
                    ]
                ),
                "delta_nearest_landmark_switch_rate_ppo_minus_random": float(
                    ppo_row["nearest_landmark_switch_rate_mean"]
                    - random_row["nearest_landmark_switch_rate_mean"]
                ),
            }
        )

    return pd.DataFrame(rows)


def paired_plot(
    seed_summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Create a paired per-seed random-versus-PPO comparison plot."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for training_seed, group in seed_summary.groupby("training_seed"):
        values = group.set_index("policy")[metric]
        if {"random", "ppo"}.issubset(values.index):
            ax.plot(
                [0, 1],
                [values["random"], values["ppo"]],
                marker="o",
                alpha=0.75,
                label=f"seed {training_seed}",
            )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Random policy", "Trained PPO"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def choose_representative_seed(
    profile: str,
    fallback_seeds: list[int],
) -> int:
    """
    Choose the median seed by the existing final assignment-distance result.

    This avoids selecting only the best or worst trained policy.
    """
    summary_path = PROJECT_ROOT / "results" / "multi_agent_seed_results.csv"
    if not summary_path.exists():
        return fallback_seeds[len(fallback_seeds) // 2]

    frame = pd.read_csv(summary_path)
    frame = frame[
        (frame["profile"] == profile)
        & (frame["algorithm"] == "PPO")
    ].sort_values("final_mean_assignment_distance")

    if frame.empty:
        return fallback_seeds[len(fallback_seeds) // 2]

    return int(frame.iloc[len(frame) // 2]["seed"])


def trajectory_plot(
    trajectory_df: pd.DataFrame,
    episode_result: dict[str, Any],
    output_path: Path,
) -> None:
    """Plot agent paths, fixed landmarks, and final assignment links."""
    if trajectory_df.empty:
        raise ValueError("Trajectory data was empty.")

    policy = str(trajectory_df["policy"].iloc[0])
    episode_seed = int(trajectory_df["episode_seed"].iloc[0])

    fig, ax = plt.subplots(figsize=(7.2, 7.2))

    agents_df = trajectory_df[
        trajectory_df["entity_type"] == "agent"
    ]
    landmarks_df = (
        trajectory_df[
            trajectory_df["entity_type"] == "landmark"
        ]
        .sort_values("step")
        .drop_duplicates("entity_id")
    )

    final_agent_rows = []
    for agent, group in agents_df.groupby("entity_id"):
        group = group.sort_values("step")
        ax.plot(group["x"], group["y"], marker=".", label=agent)
        ax.scatter(
            [group.iloc[0]["x"]],
            [group.iloc[0]["y"]],
            marker="o",
            s=80,
        )
        ax.scatter(
            [group.iloc[-1]["x"]],
            [group.iloc[-1]["y"]],
            marker="^",
            s=90,
        )
        final_agent_rows.append(group.iloc[-1])

    ax.scatter(
        landmarks_df["x"],
        landmarks_df["y"],
        marker="X",
        s=150,
        label="landmarks",
    )

    landmark_lookup = {
        int(row["assigned_landmark"]): (
            float(row["x"]),
            float(row["y"]),
        )
        for _, row in landmarks_df.iterrows()
    }

    for landmark_index, (x_value, y_value) in landmark_lookup.items():
        ax.annotate(
            f"L{landmark_index}",
            (x_value, y_value),
            xytext=(5, 5),
            textcoords="offset points",
        )

    for row in final_agent_rows:
        assigned = int(row["assigned_landmark"])
        landmark_x, landmark_y = landmark_lookup[assigned]
        ax.plot(
            [float(row["x"]), landmark_x],
            [float(row["y"]), landmark_y],
            linestyle="--",
            alpha=0.45,
        )

    title_policy = "Random policy" if policy == "random" else "Trained PPO"
    ax.set_title(
        f"Simple Spread trajectory: {title_policy}\n"
        f"episode seed {episode_seed} | "
        f"return {episode_result['mean_agent_return']:.2f} | "
        f"final assignment {episode_result['final_assignment_distance']:.3f} | "
        f"collisions/cycle {episode_result['collision_pairs_per_cycle']:.3f}"
    )
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_markdown_summary(
    overall_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    representative_seed: int,
    trajectory_episode_seed: int,
    output_path: Path,
) -> None:
    """Write a compact, memo-ready diagnostics summary."""
    lookup = overall_df.set_index("policy")

    def value(policy: str, metric: str) -> tuple[float, float]:
        return (
            float(lookup.loc[policy, f"{metric}_across_seed_mean"]),
            float(lookup.loc[policy, f"{metric}_across_seed_std"]),
        )

    random_return = value("random", "mean_agent_return_mean")
    ppo_return = value("ppo", "mean_agent_return_mean")
    random_distance = value("random", "final_assignment_distance_mean")
    ppo_distance = value("ppo", "final_assignment_distance_mean")
    random_collision = value("random", "collision_pairs_per_cycle_mean")
    ppo_collision = value("ppo", "collision_pairs_per_cycle_mean")

    content = f"""# Multi-Agent Random Baseline and Trajectory Diagnostic

## Design

The diagnostic re-evaluated each of the five trained parameter-sharing PPO
policies against a random-action policy. PPO and random actions were tested on
the same 10 held-out episode seeds within each training seed. The five
training-seed means, rather than all 50 episodes, were treated as the main
replication units.

## Main Results

| Metric | Random policy, mean ± SD across seeds | Trained PPO, mean ± SD across seeds |
|---|---:|---:|
| Mean agent return | {random_return[0]:.3f} ± {random_return[1]:.3f} | {ppo_return[0]:.3f} ± {ppo_return[1]:.3f} |
| Final assignment distance | {random_distance[0]:.3f} ± {random_distance[1]:.3f} | {ppo_distance[0]:.3f} ± {ppo_distance[1]:.3f} |
| Collision pairs per cycle | {random_collision[0]:.3f} ± {random_collision[1]:.3f} | {ppo_collision[0]:.3f} ± {ppo_collision[1]:.3f} |

PPO-minus-random differences were also saved per training seed in
`multi_agent_diagnostics_paired_deltas.csv`. Positive return differences favor
PPO. Negative assignment-distance and collision differences favor PPO.

## Trajectory Inspection

The representative trajectory used training seed {representative_seed}, chosen
as the median seed by the original standard-run final assignment distance. The
matched random and PPO trajectories use the same held-out episode seed
({trajectory_episode_seed}), reducing the risk that the visual comparison is
driven only by different initial conditions.

## Interpretation Boundary

This diagnostic tests whether the trained parameter-sharing PPO baseline
outperformed random actions under the same Simple Spread initial conditions. It
does not establish robust multi-robot coordination, role specialization, or
transfer to fielded robots. The 0.25 coordination threshold remains an
analyst-defined diagnostic rather than an official environment success
criterion.
"""
    output_path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--profile",
        default="standard",
        choices=["smoke", "standard", "extended"],
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Episodes per policy and training seed. Defaults to profile eval_episodes.",
    )
    parser.add_argument(
        "--trajectory-seed",
        type=int,
        default=None,
        help="Training seed for the representative trajectory. Default: median seed.",
    )
    parser.add_argument(
        "--trajectory-episode-index",
        type=int,
        default=0,
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    seeds = args.seeds or list(config["project"]["seeds"])
    n_episodes = (
        args.episodes
        if args.episodes is not None
        else int(
            config["profiles"][args.profile]["multi_agent"][
                "eval_episodes"
            ]
        )
    )

    results_dir = PROJECT_ROOT / "results"
    plots_dir = PROJECT_ROOT / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Evaluating {len(seeds)} trained seeds, "
        f"{n_episodes} matched episodes per policy and seed."
    )
    episode_df = evaluate_all(
        config=config,
        profile=args.profile,
        seeds=seeds,
        n_episodes=n_episodes,
        device=args.device,
    )
    seed_summary = create_seed_summary(episode_df)
    overall_summary = create_overall_summary(seed_summary)
    paired_deltas = create_paired_seed_deltas(seed_summary)

    episode_df.to_csv(
        results_dir / "multi_agent_diagnostics_episode_results.csv",
        index=False,
    )
    seed_summary.to_csv(
        results_dir / "multi_agent_diagnostics_seed_summary.csv",
        index=False,
    )
    overall_summary.to_csv(
        results_dir / "multi_agent_diagnostics_overall_summary.csv",
        index=False,
    )
    paired_deltas.to_csv(
        results_dir / "multi_agent_diagnostics_paired_deltas.csv",
        index=False,
    )

    write_json(
        results_dir / "multi_agent_diagnostics_metadata.json",
        {
            "profile": args.profile,
            "training_seeds": seeds,
            "episodes_per_policy_per_training_seed": n_episodes,
            "episode_seed_rule": "900000 + training_seed * 100 + episode_index",
            "random_and_ppo_use_matched_episode_seeds": True,
            "ppo_evaluation_is_deterministic": True,
            "software": system_metadata(),
        },
    )

    paired_plot(
        seed_summary,
        metric="mean_agent_return_mean",
        ylabel="Mean agent return (higher is better)",
        title="Simple Spread: Random Policy vs Trained PPO Return",
        output_path=plots_dir
        / "simple_spread_random_vs_ppo_return.png",
    )
    paired_plot(
        seed_summary,
        metric="final_assignment_distance_mean",
        ylabel="Final assignment distance (lower is better)",
        title="Simple Spread: Random Policy vs Trained PPO Assignment Distance",
        output_path=plots_dir
        / "simple_spread_random_vs_ppo_assignment_distance.png",
    )
    paired_plot(
        seed_summary,
        metric="collision_pairs_per_cycle_mean",
        ylabel="Collision pairs per cycle (lower is better)",
        title="Simple Spread: Random Policy vs Trained PPO Collisions",
        output_path=plots_dir
        / "simple_spread_random_vs_ppo_collisions.png",
    )

    representative_seed = (
        args.trajectory_seed
        if args.trajectory_seed is not None
        else choose_representative_seed(args.profile, seeds)
    )
    trajectory_episode_seed = paired_episode_seed(
        representative_seed,
        args.trajectory_episode_index,
    )
    representative_model = PPO.load(
        str(model_path_for_seed(representative_seed, args.profile)),
        device=args.device,
    )

    trajectory_results: dict[str, dict[str, Any]] = {}
    for policy_kind in ("random", "ppo"):
        result, trajectory_df = rollout_episode(
            policy_kind=policy_kind,
            env_cfg=config["multi_agent"],
            episode_seed=trajectory_episode_seed,
            model=representative_model if policy_kind == "ppo" else None,
            record_trajectory=True,
        )
        trajectory_results[policy_kind] = result
        trajectory_df.to_csv(
            results_dir
            / (
                f"simple_spread_trajectory_seed{representative_seed}_"
                f"episode{args.trajectory_episode_index}_{policy_kind}.csv"
            ),
            index=False,
        )
        trajectory_plot(
            trajectory_df,
            episode_result=result,
            output_path=plots_dir
            / (
                f"simple_spread_trajectory_seed{representative_seed}_"
                f"episode{args.trajectory_episode_index}_{policy_kind}.png"
            ),
        )

    write_json(
        results_dir / "multi_agent_trajectory_summary.json",
        {
            "representative_training_seed": representative_seed,
            "trajectory_episode_index": args.trajectory_episode_index,
            "trajectory_episode_seed": trajectory_episode_seed,
            "selection_rule": (
                "Median training seed by original standard-run final "
                "assignment distance unless --trajectory-seed is provided."
            ),
            "random": trajectory_results["random"],
            "ppo": trajectory_results["ppo"],
        },
    )

    write_markdown_summary(
        overall_df=overall_summary,
        paired_df=paired_deltas,
        representative_seed=representative_seed,
        trajectory_episode_seed=trajectory_episode_seed,
        output_path=results_dir
        / "multi_agent_diagnostics_summary.md",
    )

    print("\nDiagnostics complete.")
    for path in [
        results_dir / "multi_agent_diagnostics_episode_results.csv",
        results_dir / "multi_agent_diagnostics_seed_summary.csv",
        results_dir / "multi_agent_diagnostics_overall_summary.csv",
        results_dir / "multi_agent_diagnostics_paired_deltas.csv",
        results_dir / "multi_agent_diagnostics_summary.md",
        plots_dir / "simple_spread_random_vs_ppo_return.png",
        plots_dir / "simple_spread_random_vs_ppo_assignment_distance.png",
        plots_dir / "simple_spread_random_vs_ppo_collisions.png",
    ]:
        print(" -", path)

    print(
        f"Representative trajectory seed: {representative_seed}; "
        f"episode seed: {trajectory_episode_seed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
