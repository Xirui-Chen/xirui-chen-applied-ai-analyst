#!/usr/bin/env python3
"""Create Week 5 learning-curve and comparison plots with Matplotlib."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import PROJECT_ROOT


def collect_histories(base: Path, kind: str) -> pd.DataFrame:
    frames = []
    for history_path in sorted(base.glob("*/evaluation_history.csv")):
        summary_path = history_path.parent / "run_summary.json"
        if not summary_path.exists():
            continue
        frame = pd.read_csv(history_path)
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        frame["run_id"] = summary["run_id"]
        frame["seed"] = summary["seed"]
        frame["profile"] = summary["profile"]
        frame["algorithm"] = summary["algorithm"]
        frame["environment"] = (
            summary["environment_requested"] if kind == "single" else summary["environment"]
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_single_learning_curves(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return
    for (profile, environment), subset in df.groupby(["profile", "environment"]):
        fig, ax = plt.subplots(figsize=(10, 6))
        for algorithm, algo_df in subset.groupby("algorithm"):
            stats = algo_df.groupby("timesteps")["mean_reward"].agg(["mean", "std"]).reset_index()
            ax.plot(stats["timesteps"], stats["mean"], label=algorithm)
            ax.fill_between(
                stats["timesteps"],
                stats["mean"] - stats["std"].fillna(0),
                stats["mean"] + stats["std"].fillna(0),
                alpha=0.2,
            )
        ax.set_title(f"{environment}: PPO vs SAC Evaluation Learning Curves ({profile})")
        ax.set_xlabel("Environment timesteps")
        ax.set_ylabel("Deterministic evaluation return")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        filename = f"{environment}_{profile}_ppo_sac_learning_curves.png".replace("/", "_")
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def plot_final_returns(output_dir: Path) -> None:
    csv_path = PROJECT_ROOT / "results" / "single_agent_seed_results.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    for (profile, environment), subset in df.groupby(["profile", "environment_requested"]):
        fig, ax = plt.subplots(figsize=(9, 5))
        algorithms = sorted(subset["algorithm"].unique())
        positions = {algorithm: i for i, algorithm in enumerate(algorithms)}
        for algorithm, group in subset.groupby("algorithm"):
            x = np.full(len(group), positions[algorithm], dtype=float)
            ax.scatter(x, group["final_eval_mean_reward"], s=55)
            ax.hlines(
                group["final_eval_mean_reward"].mean(),
                positions[algorithm] - 0.20,
                positions[algorithm] + 0.20,
                linewidth=3,
            )
        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels(algorithms)
        ax.set_ylabel("Final deterministic evaluation return")
        ax.set_title(f"{environment}: Final Return Across Seeds ({profile})")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        filename = f"{environment}_{profile}_final_return_by_seed.png".replace("/", "_")
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def plot_wallclock_tradeoff(output_dir: Path) -> None:
    csv_path = PROJECT_ROOT / "results" / "single_agent_seed_results.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    for profile, subset in df.groupby("profile"):
        fig, ax = plt.subplots(figsize=(9, 6))
        for (environment, algorithm), group in subset.groupby(["environment_requested", "algorithm"]):
            ax.scatter(
                group["wallclock_sec"] / 60.0,
                group["final_eval_mean_reward"],
                label=f"{environment} | {algorithm}",
                s=55,
            )
        ax.set_xlabel("Wall-clock training time (minutes)")
        ax.set_ylabel("Final deterministic evaluation return")
        ax.set_title(f"Wall-Clock Cost vs Final Return ({profile})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"single_agent_{profile}_wallclock_tradeoff.png", dpi=180)
        plt.close(fig)


def plot_multi_learning_curves(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return
    for profile, subset in df.groupby("profile"):
        stats = (
            subset.groupby("timesteps")
            .agg(
                return_mean=("mean_agent_return", "mean"),
                return_std=("mean_agent_return", "std"),
                distance_mean=("mean_final_assignment_distance", "mean"),
                distance_std=("mean_final_assignment_distance", "std"),
            )
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(stats["timesteps"], stats["return_mean"])
        ax.fill_between(
            stats["timesteps"],
            stats["return_mean"] - stats["return_std"].fillna(0),
            stats["return_mean"] + stats["return_std"].fillna(0),
            alpha=0.2,
        )
        ax.set_xlabel("Environment timesteps")
        ax.set_ylabel("Mean agent return")
        ax.set_title(f"Simple Spread Parameter-Sharing PPO Learning Curve ({profile})")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"simple_spread_{profile}_learning_curve.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(stats["timesteps"], stats["distance_mean"])
        ax.fill_between(
            stats["timesteps"],
            stats["distance_mean"] - stats["distance_std"].fillna(0),
            stats["distance_mean"] + stats["distance_std"].fillna(0),
            alpha=0.2,
        )
        ax.set_xlabel("Environment timesteps")
        ax.set_ylabel("Final assignment distance (lower is better)")
        ax.set_title(f"Simple Spread Coordination Distance Across Seeds ({profile})")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"simple_spread_{profile}_coordination_distance.png", dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "plots"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    single = collect_histories(PROJECT_ROOT / "results" / "raw" / "single_agent", kind="single")
    multi = collect_histories(PROJECT_ROOT / "results" / "raw" / "multi_agent", kind="multi")

    plot_single_learning_curves(single, output_dir)
    plot_final_returns(output_dir)
    plot_wallclock_tradeoff(output_dir)
    plot_multi_learning_curves(multi, output_dir)

    print("Plot generation complete.")
    for file in sorted(output_dir.glob("*.png")):
        print(" -", file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
