#!/usr/bin/env python3
"""Validate the compact sample-data contracts before Streamlit starts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "sample_data"

REQUIRED = {
    "fleet_unit_status.csv": {
        "unit_id",
        "status",
        "battery_soc_pct",
        "active_alert",
        "alert_severity_score",
        "recommended_action",
    },
    "fleet_telemetry_5min.csv": {
        "timestamp",
        "robot_id",
        "battery_soc_pct",
        "motor_temp_c",
        "wifi_rssi_dbm",
        "task_success_probability",
    },
    "anomaly_metrics_summary.csv": {
        "scope",
        "method",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "auroc_mean",
    },
    "rl_single_agent_summary.csv": {
        "environment",
        "algorithm",
        "final_reward_mean",
        "wallclock_sec_mean",
        "threshold_success_rate",
    },
    "rl_learning_curves.csv": {
        "environment",
        "algorithm",
        "seed",
        "timesteps",
        "mean_reward",
    },
    "decision_judgments.csv": {
        "scenario_id",
        "domain",
        "judge_seed",
        "verdict",
        "parse_ok",
    },
    "decision_reliability_summary.csv": {
        "scope",
        "domain",
        "krippendorff_alpha_nominal",
    },
    "decision_scenario_results.csv": {
        "scenario_id",
        "domain",
        "candidate_action",
        "target_action",
        "exact_match",
    },
}


def main() -> int:
    errors: list[str] = []

    for filename, expected_columns in REQUIRED.items():
        path = DATA / filename
        if not path.exists():
            errors.append(f"Missing file: {filename}")
            continue
        frame = pd.read_csv(path)
        missing = expected_columns - set(frame.columns)
        if missing:
            errors.append(f"{filename}: missing columns {sorted(missing)}")

    if errors:
        raise SystemExit("Data validation failed:\n- " + "\n- ".join(errors))

    fleet = pd.read_csv(DATA / "fleet_unit_status.csv")
    judgments = pd.read_csv(DATA / "decision_judgments.csv")
    scenarios = pd.read_csv(DATA / "decision_scenario_results.csv")
    manifest = json.loads(
        (DATA / "data_manifest.json").read_text(encoding="utf-8")
    )

    assertions = [
        (fleet["unit_id"].nunique() == 5, "Expected five fleet units"),
        (len(judgments) == 150, "Expected 150 seeded judgments"),
        (judgments["scenario_id"].nunique() == 50, "Expected 50 scenarios"),
        (set(judgments["judge_seed"]) == {7, 17, 27}, "Expected seeds 7, 17, 27"),
        (len(scenarios) == 50, "Expected 50 scenario-level rows"),
        (bool(judgments["parse_ok"].astype(bool).all()), "Expected all judgments to parse"),
        (manifest.get("public_safe") is True, "Manifest must be public-safe"),
    ]
    failed = [message for passed, message in assertions if not passed]
    if failed:
        raise SystemExit("Data assertions failed:\n- " + "\n- ".join(failed))

    print("PASS: dashboard sample data validated.")
    print(f"  Fleet units: {fleet['unit_id'].nunique()}")
    print(f"  Decision scenarios: {judgments['scenario_id'].nunique()}")
    print(f"  Seeded judgments: {len(judgments)}")
    print(f"  Sample data files: {len(manifest['files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
