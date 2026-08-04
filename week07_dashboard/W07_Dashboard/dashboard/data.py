"""Cached data access for the Week 7 dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "sample_data"


@st.cache_data(show_spinner=False)
def read_csv(name: str, parse_dates: tuple[str, ...] = ()) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Required dashboard data is missing: {path}")
    frame = pd.read_csv(path)
    for column in parse_dates:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


@st.cache_data(show_spinner=False)
def read_manifest() -> dict[str, Any]:
    return json.loads((DATA_DIR / "data_manifest.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> dict[str, Any]:
    return {
        "anomaly_summary": read_csv("anomaly_metrics_summary.csv"),
        "anomaly_per_split": read_csv("anomaly_metrics_per_split.csv"),
        "anomaly_wilcoxon": read_csv("anomaly_wilcoxon_auroc.csv"),
        "alert_history": read_csv(
            "fleet_alert_history.csv",
            parse_dates=("event_time", "resolved_time"),
        ),
        "fleet_telemetry": read_csv(
            "fleet_telemetry_5min.csv",
            parse_dates=("timestamp",),
        ),
        "fleet_status": read_csv(
            "fleet_unit_status.csv",
            parse_dates=("last_seen",),
        ),
        "rl_summary": read_csv("rl_single_agent_summary.csv"),
        "rl_seed": read_csv("rl_single_agent_seed_results.csv"),
        "rl_efficiency": read_csv("rl_sample_efficiency.csv"),
        "rl_wallclock": read_csv("rl_wallclock.csv"),
        "rl_learning": read_csv("rl_learning_curves.csv"),
        "rl_multi_summary": read_csv("rl_multi_agent_summary.csv"),
        "rl_multi_seed": read_csv("rl_multi_agent_seed_results.csv"),
        "rl_multi_diag": read_csv("rl_multi_agent_diagnostics.csv"),
        "rl_multi_diag_seed": read_csv(
            "rl_multi_agent_diagnostics_by_seed.csv"
        ),
        "rl_multi_deltas": read_csv("rl_multi_agent_paired_deltas.csv"),
        "decision_domain": read_csv("decision_domain_summary.csv"),
        "decision_reliability": read_csv(
            "decision_reliability_summary.csv"
        ),
        "decision_kappa": read_csv("decision_pairwise_kappa.csv"),
        "decision_scenarios": read_csv("decision_scenario_results.csv"),
        "decision_judgments": read_csv("decision_judgments.csv"),
        "decision_rules": read_csv("decision_rule_predictions.csv"),
        "decision_errors": read_csv("decision_error_analysis.csv"),
        "manifest": read_manifest(),
    }
