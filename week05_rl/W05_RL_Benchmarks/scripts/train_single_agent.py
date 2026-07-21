#!/usr/bin/env python3
"""Train one PPO or SAC baseline run on one Gymnasium environment."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from common import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    ensure_output_dirs,
    load_config,
    make_single_agent_env,
    normalized_curve_auc,
    reward_at_fraction,
    slugify,
    system_metadata,
    write_json,
)


class TimedEvalCallback(BaseCallback):
    """Periodically evaluate on a separate environment and persist results."""

    def __init__(
        self,
        eval_env,
        eval_freq: int,
        n_eval_episodes: int,
        csv_path: Path,
        reward_threshold: float,
        deterministic: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.csv_path = csv_path
        self.reward_threshold = float(reward_threshold)
        self.deterministic = deterministic
        self.rows: list[dict[str, Any]] = []
        self.start_time = 0.0
        self.next_eval = self.eval_freq
        self.first_threshold_timestep: int | None = None

    def _on_training_start(self) -> None:
        self.start_time = time.perf_counter()

    def _evaluate(self) -> None:
        returns, lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            return_episode_rewards=True,
            warn=True,
        )
        elapsed = time.perf_counter() - self.start_time
        mean_reward = float(np.mean(returns))
        row = {
            "timesteps": int(self.num_timesteps),
            "mean_reward": mean_reward,
            "std_reward": float(np.std(returns, ddof=0)),
            "median_reward": float(np.median(returns)),
            "min_reward": float(np.min(returns)),
            "max_reward": float(np.max(returns)),
            "mean_episode_length": float(np.mean(lengths)),
            "wallclock_sec": elapsed,
            "environment_steps_per_sec": float(self.num_timesteps / max(elapsed, 1e-9)),
            "n_eval_episodes": self.n_eval_episodes,
        }
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.csv_path, index=False)

        self.logger.record("eval/mean_reward", row["mean_reward"])
        self.logger.record("eval/std_reward", row["std_reward"])
        self.logger.record("eval/wallclock_sec", row["wallclock_sec"])

        if mean_reward >= self.reward_threshold and self.first_threshold_timestep is None:
            self.first_threshold_timestep = int(self.num_timesteps)

        if self.verbose:
            print(
                f"[eval] steps={self.num_timesteps:,} "
                f"reward={mean_reward:.2f} ± {row['std_reward']:.2f} "
                f"elapsed={elapsed:.1f}s"
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


def build_model(
    algorithm: str,
    policy: str,
    env,
    hyperparameters: dict[str, Any],
    seed: int,
    device: str,
    tensorboard_dir: Path,
    verbose: int,
):
    """Instantiate the requested Stable-Baselines3 algorithm."""
    kwargs = dict(hyperparameters)
    kwargs.pop("policy", None)
    common_kwargs = {
        "policy": policy,
        "env": env,
        "seed": seed,
        "device": device,
        "tensorboard_log": str(tensorboard_dir),
        "verbose": verbose,
    }
    if algorithm == "PPO":
        return PPO(**common_kwargs, **kwargs)
    if algorithm == "SAC":
        return SAC(**common_kwargs, **kwargs)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--environment",
        required=True,
        choices=["LunarLanderContinuous-v3", "BipedalWalker-v3"],
    )
    parser.add_argument("--algorithm", required=True, choices=["PPO", "SAC"])
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

    profile_cfg = config["profiles"][args.profile]["single_agent"][args.environment]
    env_cfg = config["single_agent"]["environments"][args.environment]
    algo_cfg = config["single_agent"]["algorithms"][args.algorithm]

    total_timesteps = int(profile_cfg["total_timesteps"])
    eval_freq = int(profile_cfg["eval_freq"])
    eval_episodes = int(profile_cfg["eval_episodes"])
    solved_reward = float(env_cfg["solved_reward"])
    device = args.device or config["project"]["device"]

    run_id = slugify(f"{args.environment}__{args.algorithm}__seed{args.seed}__{args.profile}")
    run_dir = PROJECT_ROOT / "results" / "raw" / "single_agent" / run_id
    summary_path = run_dir / "run_summary.json"

    if summary_path.exists() and not args.overwrite:
        print(f"SKIP: completed run exists at {summary_path}")
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = PROJECT_ROOT / "tensorboard" / "single_agent" / run_id
    model_path = PROJECT_ROOT / "models" / "single_agent" / run_id / "model"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    train_env, actual_env_id = make_single_agent_env(
        args.environment,
        env_cfg,
        seed=args.seed,
        monitor_path=run_dir / "train.monitor.csv",
    )
    eval_env, _ = make_single_agent_env(
        args.environment,
        env_cfg,
        seed=args.seed + 100_000,
        monitor_path=run_dir / "eval.monitor.csv",
    )

    callback = TimedEvalCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        csv_path=run_dir / "evaluation_history.csv",
        reward_threshold=solved_reward,
        deterministic=bool(config["project"]["deterministic_evaluation"]),
        verbose=args.verbose,
    )

    model = build_model(
        algorithm=args.algorithm,
        policy=algo_cfg["policy"],
        env=train_env,
        hyperparameters=algo_cfg,
        seed=args.seed,
        device=device,
        tensorboard_dir=tensorboard_dir,
        verbose=args.verbose,
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
        rewards = eval_df["mean_reward"].to_numpy(dtype=float)

        summary = {
            "status": "completed",
            "run_id": run_id,
            "profile": args.profile,
            "environment_requested": args.environment,
            "environment_actual": actual_env_id,
            "algorithm": args.algorithm,
            "seed": args.seed,
            "total_timesteps_requested": total_timesteps,
            "total_timesteps_actual": int(model.num_timesteps),
            "evaluation_frequency": eval_freq,
            "evaluation_episodes": eval_episodes,
            "deterministic_evaluation": bool(config["project"]["deterministic_evaluation"]),
            "solved_reward_threshold": solved_reward,
            "timesteps_to_threshold": callback.first_threshold_timestep,
            "threshold_reached": callback.first_threshold_timestep is not None,
            "final_eval_mean_reward": float(rewards[-1]),
            "final_eval_std_reward": float(eval_df["std_reward"].iloc[-1]),
            "best_eval_mean_reward": float(np.max(rewards)),
            "best_eval_timestep": int(timesteps[int(np.argmax(rewards))]),
            "normalized_learning_curve_auc": normalized_curve_auc(timesteps, rewards),
            "reward_at_25pct_budget": reward_at_fraction(timesteps, rewards, 0.25, total_timesteps),
            "reward_at_50pct_budget": reward_at_fraction(timesteps, rewards, 0.50, total_timesteps),
            "reward_at_75pct_budget": reward_at_fraction(timesteps, rewards, 0.75, total_timesteps),
            "wallclock_sec": wallclock_sec,
            "environment_steps_per_sec": float(model.num_timesteps / max(wallclock_sec, 1e-9)),
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
        eval_env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
