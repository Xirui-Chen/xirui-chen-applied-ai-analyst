#!/usr/bin/env python3
"""Aggregate completed Week 5 run summaries into reproducible CSV tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import PROJECT_ROOT


def load_json_summaries(path: Path) -> pd.DataFrame:
    rows = []
    for file in sorted(path.glob("*/run_summary.json")):
        with file.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        row["summary_file"] = str(file.relative_to(PROJECT_ROOT))
        rows.append(row)
    return pd.DataFrame(rows)


def safe_cv(std_value: float, mean_value: float) -> float:
    if not np.isfinite(std_value) or not np.isfinite(mean_value):
        return float("nan")
    return float(std_value / (abs(mean_value) + 1e-9))


def aggregate_single(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        print("No completed single-agent summaries found.")
        return

    flat = pd.json_normalize(df.to_dict(orient="records"))
    flat.to_csv(output_dir / "single_agent_seed_results.csv", index=False)

    records = []
    for keys, group in flat.groupby(
        ["profile", "environment_requested", "algorithm"],
        dropna=False,
    ):
        final_mean = float(group["final_eval_mean_reward"].mean())
        final_std = float(group["final_eval_mean_reward"].std(ddof=1))
        records.append(
            {
                "profile": keys[0],
                "environment": keys[1],
                "algorithm": keys[2],
                "n_seeds": len(group),
                "final_reward_mean": final_mean,
                "final_reward_std_across_seeds": final_std,
                "convergence_stability_cv": safe_cv(final_std, final_mean),
                "best_reward_mean": float(group["best_eval_mean_reward"].mean()),
                "best_reward_std": float(group["best_eval_mean_reward"].std(ddof=1)),
                "learning_curve_auc_mean": float(group["normalized_learning_curve_auc"].mean()),
                "learning_curve_auc_std": float(group["normalized_learning_curve_auc"].std(ddof=1)),
                "wallclock_sec_mean": float(group["wallclock_sec"].mean()),
                "wallclock_sec_std": float(group["wallclock_sec"].std(ddof=1)),
                "steps_per_sec_mean": float(group["environment_steps_per_sec"].mean()),
                "threshold_success_rate": float(group["threshold_reached"].mean()),
                "median_timesteps_to_threshold": float(
                    group.loc[group["threshold_reached"], "timesteps_to_threshold"].median()
                )
                if group["threshold_reached"].any()
                else float("nan"),
            }
        )

    pd.DataFrame(records).to_csv(output_dir / "single_agent_summary.csv", index=False)

    sample_cols = [
        "profile",
        "environment_requested",
        "algorithm",
        "seed",
        "reward_at_25pct_budget",
        "reward_at_50pct_budget",
        "reward_at_75pct_budget",
        "final_eval_mean_reward",
        "normalized_learning_curve_auc",
        "threshold_reached",
        "timesteps_to_threshold",
    ]
    flat[sample_cols].to_csv(output_dir / "sample_efficiency_summary.csv", index=False)

    wallclock_cols = [
        "profile",
        "environment_requested",
        "algorithm",
        "seed",
        "total_timesteps_actual",
        "wallclock_sec",
        "environment_steps_per_sec",
    ]
    flat[wallclock_cols].to_csv(output_dir / "wallclock_summary.csv", index=False)


def aggregate_multi(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        print("No completed multi-agent summaries found.")
        return

    flat = pd.json_normalize(df.to_dict(orient="records"))
    flat.to_csv(output_dir / "multi_agent_seed_results.csv", index=False)

    records = []
    for keys, group in flat.groupby(["profile", "environment", "algorithm"], dropna=False):
        return_mean = float(group["final_mean_agent_return"].mean())
        return_std = float(group["final_mean_agent_return"].std(ddof=1))
        records.append(
            {
                "profile": keys[0],
                "environment": keys[1],
                "algorithm": keys[2],
                "n_seeds": len(group),
                "final_mean_agent_return_mean": return_mean,
                "final_mean_agent_return_std_across_seeds": return_std,
                "return_stability_cv": safe_cv(return_std, return_mean),
                "learning_curve_auc_mean": float(group["normalized_learning_curve_auc"].mean()),
                "learning_curve_auc_std": float(group["normalized_learning_curve_auc"].std(ddof=1)),
                "final_assignment_distance_mean": float(group["final_mean_assignment_distance"].mean()),
                "final_assignment_distance_std": float(group["final_mean_assignment_distance"].std(ddof=1)),
                "coordination_success_rate_mean": float(group["final_coordination_success_rate"].mean()),
                "collision_pairs_per_cycle_mean": float(group["final_collision_pairs_per_cycle"].mean()),
                "wallclock_sec_mean": float(group["wallclock_sec"].mean()),
                "wallclock_sec_std": float(group["wallclock_sec"].std(ddof=1)),
                "steps_per_sec_mean": float(group["environment_steps_per_sec"].mean()),
            }
        )

    pd.DataFrame(records).to_csv(output_dir / "multi_agent_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    single = load_json_summaries(PROJECT_ROOT / "results" / "raw" / "single_agent")
    multi = load_json_summaries(PROJECT_ROOT / "results" / "raw" / "multi_agent")

    aggregate_single(single, output_dir)
    aggregate_multi(multi, output_dir)

    print("Aggregation complete.")
    for file in sorted(output_dir.glob("*.csv")):
        print(" -", file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
