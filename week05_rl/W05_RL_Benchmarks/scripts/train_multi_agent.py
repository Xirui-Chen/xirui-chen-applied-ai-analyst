#!/usr/bin/env python3
"""Train a parameter-sharing PPO baseline on Simple Spread."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import supersuit as ss
from scipy.optimize import linear_sum_assignment
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

from common import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    ensure_output_dirs,
    load_config,
    normalized_curve_auc,
    reward_at_fraction,
    slugify,
    system_metadata,
    write_json,
)


def import_simple_spread():
    """Use current MPE2, with a legacy PettingZoo fallback."""
    try:
        from mpe2 import simple_spread_v3
        return simple_spread_v3, "mpe2.simple_spread_v3"
    except ImportError:
        from pettingzoo.mpe import simple_spread_v3
        return simple_spread_v3, "pettingzoo.mpe.simple_spread_v3"


def make_parallel_env(cfg: dict[str, Any], seed: int):
    module, source = import_simple_spread()
    env = module.parallel_env(
        N=int(cfg["n_agents"]),
        local_ratio=float(cfg["local_ratio"]),
        max_cycles=int(cfg["max_cycles"]),
        continuous_actions=bool(cfg["continuous_actions"]),
    )
    env.reset(seed=seed)
    return env, source


def make_sb3_vec_env(cfg: dict[str, Any], seed: int):
    parallel_env, source = make_parallel_env(cfg, seed)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(parallel_env)
    vec_env = ss.concat_vec_envs_v1(
        vec_env,
        int(cfg["vector_copies"]),
        num_cpus=0,
        base_class="stable_baselines3",
    )
    return vec_env, source


def coordination_state_metrics(
    observations: dict[str, np.ndarray],
    n_agents: int,
) -> tuple[float, int]:
    """Estimate landmark coverage and collisions from Simple Spread observations."""
    agents = list(observations)
    if not agents:
        return float("nan"), 0

    agent_positions = np.vstack(
        [np.asarray(observations[a], dtype=float)[2:4] for a in agents]
    )
    first_obs = np.asarray(observations[agents[0]], dtype=float)
    first_pos = first_obs[2:4]
    landmark_rel = first_obs[4 : 4 + 2 * n_agents].reshape(n_agents, 2)
    landmark_positions = first_pos + landmark_rel

    distances = np.linalg.norm(
        agent_positions[:, None, :] - landmark_positions[None, :, :],
        axis=2,
    )
    rows, cols = linear_sum_assignment(distances)
    mean_assignment_distance = float(np.mean(distances[rows, cols]))

    collision_pairs = 0
    for i in range(len(agent_positions)):
        for j in range(i + 1, len(agent_positions)):
            if np.linalg.norm(agent_positions[i] - agent_positions[j]) < 0.30:
                collision_pairs += 1

    return mean_assignment_distance, collision_pairs


def evaluate_shared_policy(
    model,
    env_cfg: dict[str, Any],
    n_episodes: int,
    base_seed: int,
    deterministic: bool,
) -> pd.DataFrame:
    """Evaluate a shared policy on fresh parallel-environment episodes."""
    rows = []
    n_agents = int(env_cfg["n_agents"])
    success_distance = float(env_cfg["coordination_success_distance"])

    for episode in range(n_episodes):
        env, source = make_parallel_env(env_cfg, seed=base_seed + episode)
        observations, _ = env.reset(seed=base_seed + episode)
        agent_returns = {agent: 0.0 for agent in env.possible_agents}
        assignment_distances = []
        collision_pairs_total = 0
        cycles = 0

        while env.agents:
            actions = {}
            for agent in env.agents:
                action, _ = model.predict(observations[agent], deterministic=deterministic)
                actions[agent] = action

            observations, rewards, terminations, truncations, _ = env.step(actions)
            for agent, reward in rewards.items():
                agent_returns[agent] += float(reward)

            if observations:
                assignment_distance, collision_pairs = coordination_state_metrics(
                    observations,
                    n_agents=n_agents,
                )
                assignment_distances.append(assignment_distance)
                collision_pairs_total += collision_pairs
            cycles += 1

            if all(
                terminations.get(agent, False) or truncations.get(agent, False)
                for agent in set(terminations) | set(truncations)
            ):
                break

        final_distance = float(assignment_distances[-1]) if assignment_distances else float("nan")
        rows.append(
            {
                "episode": episode,
                "mean_agent_return": float(np.mean(list(agent_returns.values()))),
                "team_return_sum": float(np.sum(list(agent_returns.values()))),
                "mean_assignment_distance": float(np.mean(assignment_distances))
                if assignment_distances
                else float("nan"),
                "final_assignment_distance": final_distance,
                "coordination_success": int(final_distance <= success_distance)
                if np.isfinite(final_distance)
                else 0,
                "collision_pairs_per_cycle": float(collision_pairs_total / max(cycles, 1)),
                "cycles": cycles,
                "environment_source": source,
            }
        )
        env.close()

    return pd.DataFrame(rows)


class MultiAgentEvalCallback(BaseCallback):
    """Evaluate team return and coordination proxies throughout training."""

    def __init__(
        self,
        env_cfg: dict[str, Any],
        eval_freq: int,
        n_eval_episodes: int,
        csv_path: Path,
        seed: int,
        deterministic: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.env_cfg = env_cfg
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.csv_path = csv_path
        self.seed = seed
        self.deterministic = deterministic
        self.rows: list[dict[str, Any]] = []
        self.start_time = 0.0
        self.next_eval = self.eval_freq
        self.first_coordination_timestep: int | None = None
        self.eval_index = 0

    def _on_training_start(self) -> None:
        self.start_time = time.perf_counter()

    def _evaluate(self) -> None:
        episode_df = evaluate_shared_policy(
            self.model,
            env_cfg=self.env_cfg,
            n_episodes=self.n_eval_episodes,
            base_seed=self.seed + 100_000 + self.eval_index * 1_000,
            deterministic=self.deterministic,
        )
        elapsed = time.perf_counter() - self.start_time
        row = {
            "timesteps": int(self.num_timesteps),
            "mean_agent_return": float(episode_df["mean_agent_return"].mean()),
            "std_agent_return": float(episode_df["mean_agent_return"].std(ddof=0)),
            "mean_team_return_sum": float(episode_df["team_return_sum"].mean()),
            "mean_assignment_distance": float(episode_df["mean_assignment_distance"].mean()),
            "mean_final_assignment_distance": float(episode_df["final_assignment_distance"].mean()),
            "coordination_success_rate": float(episode_df["coordination_success"].mean()),
            "mean_collision_pairs_per_cycle": float(episode_df["collision_pairs_per_cycle"].mean()),
            "wallclock_sec": elapsed,
            "environment_steps_per_sec": float(self.num_timesteps / max(elapsed, 1e-9)),
            "n_eval_episodes": self.n_eval_episodes,
        }
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.csv_path, index=False)
        self.eval_index += 1

        self.logger.record("eval/mean_agent_return", row["mean_agent_return"])
        self.logger.record("eval/mean_assignment_distance", row["mean_assignment_distance"])
        self.logger.record("eval/coordination_success_rate", row["coordination_success_rate"])

        threshold = float(self.env_cfg["coordination_success_distance"])
        if row["mean_final_assignment_distance"] <= threshold and self.first_coordination_timestep is None:
            self.first_coordination_timestep = int(self.num_timesteps)

        if self.verbose:
            print(
                f"[multi-eval] steps={self.num_timesteps:,} "
                f"return={row['mean_agent_return']:.2f} "
                f"assignment={row['mean_final_assignment_distance']:.3f} "
                f"success={row['coordination_success_rate']:.2f}"
            )

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_eval:
            self._evaluate()
            while self.next_eval <= self.num_timesteps:
                self.next_eval += self.eval_freq
        return True

    def _on_training_end(self) -> None:
        if not self.rows or self.rows[-1]["timesteps"] < self.num_timesteps:
            self._evaluate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--profile", default="standard", choices=["smoke", "standard", "extended"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    config = load_config(args.config)
    env_cfg = config["multi_agent"]
    algo_cfg = env_cfg["algorithm"]
    profile_cfg = config["profiles"][args.profile]["multi_agent"]

    total_timesteps = int(profile_cfg["total_timesteps"])
    eval_freq = int(profile_cfg["eval_freq"])
    eval_episodes = int(profile_cfg["eval_episodes"])
    device = args.device or config["project"]["device"]

    run_id = slugify(f"simple_spread_v3__PPO__seed{args.seed}__{args.profile}")
    run_dir = PROJECT_ROOT / "results" / "raw" / "multi_agent" / run_id
    summary_path = run_dir / "run_summary.json"

    if summary_path.exists() and not args.overwrite:
        print(f"SKIP: completed run exists at {summary_path}")
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = PROJECT_ROOT / "tensorboard" / "multi_agent" / run_id
    model_path = PROJECT_ROOT / "models" / "multi_agent" / run_id / "model"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    train_env, environment_source = make_sb3_vec_env(env_cfg, seed=args.seed)
    hyperparameters = dict(algo_cfg)
    algorithm_name = hyperparameters.pop("name")
    policy = hyperparameters.pop("policy")

    callback = MultiAgentEvalCallback(
        env_cfg=env_cfg,
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        csv_path=run_dir / "evaluation_history.csv",
        seed=args.seed,
        deterministic=bool(config["project"]["deterministic_evaluation"]),
        verbose=args.verbose,
    )

    # The raw PettingZoo environment has already been reset with args.seed.
    # Seed Python, NumPy, and PyTorch here, but do not ask SB3 to call
    # ConcatVecEnv.seed(), because that method is not implemented in some
    # SuperSuit/SB3 combinations.
    set_random_seed(args.seed)

    model = PPO(
        policy,
        train_env,
        seed=None,
        device=device,
        tensorboard_log=str(tensorboard_dir),
        verbose=args.verbose,
        **hyperparameters,
    )

    started = time.perf_counter()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=run_id,
            progress_bar=args.progress,
        )
        wallclock_sec = time.perf_counter() - started
        model.save(str(model_path))

        eval_df = pd.DataFrame(callback.rows)
        timesteps = eval_df["timesteps"].to_numpy(dtype=float)
        rewards = eval_df["mean_agent_return"].to_numpy(dtype=float)

        summary = {
            "status": "completed",
            "run_id": run_id,
            "profile": args.profile,
            "environment": "simple_spread_v3",
            "environment_source": environment_source,
            "algorithm": algorithm_name,
            "training_design": "parameter-sharing PPO via SuperSuit",
            "seed": args.seed,
            "n_agents": int(env_cfg["n_agents"]),
            "vector_copies": int(env_cfg["vector_copies"]),
            "total_timesteps_requested": total_timesteps,
            "total_timesteps_actual": int(model.num_timesteps),
            "evaluation_frequency": eval_freq,
            "evaluation_episodes": eval_episodes,
            "coordination_success_distance": float(env_cfg["coordination_success_distance"]),
            "timesteps_to_coordination_distance": callback.first_coordination_timestep,
            "coordination_distance_reached": callback.first_coordination_timestep is not None,
            "final_mean_agent_return": float(rewards[-1]),
            "final_std_agent_return": float(eval_df["std_agent_return"].iloc[-1]),
            "best_mean_agent_return": float(np.max(rewards)),
            "best_return_timestep": int(timesteps[int(np.argmax(rewards))]),
            "final_mean_assignment_distance": float(eval_df["mean_final_assignment_distance"].iloc[-1]),
            "final_coordination_success_rate": float(eval_df["coordination_success_rate"].iloc[-1]),
            "final_collision_pairs_per_cycle": float(eval_df["mean_collision_pairs_per_cycle"].iloc[-1]),
            "normalized_learning_curve_auc": normalized_curve_auc(timesteps, rewards),
            "return_at_25pct_budget": reward_at_fraction(timesteps, rewards, 0.25, total_timesteps),
            "return_at_50pct_budget": reward_at_fraction(timesteps, rewards, 0.50, total_timesteps),
            "return_at_75pct_budget": reward_at_fraction(timesteps, rewards, 0.75, total_timesteps),
            "wallclock_sec": wallclock_sec,
            "environment_steps_per_sec": float(model.num_timesteps / max(wallclock_sec, 1e-9)),
            "environment_config": {
                key: env_cfg[key]
                for key in [
                    "n_agents",
                    "local_ratio",
                    "max_cycles",
                    "continuous_actions",
                    "vector_copies",
                    "coordination_success_distance",
                ]
            },
            "hyperparameters": algo_cfg,
            "software": system_metadata(),
        }
        write_json(summary_path, summary)
        print(f"Completed: {run_id}")
        print(f"Summary: {summary_path}")
    except Exception as exc:
        write_json(
            run_dir / "run_error.json",
            {
                "status": "failed",
                "run_id": run_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "software": system_metadata(),
            },
        )
        raise
    finally:
        train_env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
