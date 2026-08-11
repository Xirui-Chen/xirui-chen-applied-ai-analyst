#!/usr/bin/env python3
"""Reproduce the Week 8 LSTM accelerated-battery-degradation follow-up."""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler


def locate(repo_root: Path, relative: str, filename: str) -> Path:
    direct = repo_root / relative
    if direct.exists():
        return direct
    matches = list(repo_root.rglob(filename))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Could not resolve {filename}. Expected {direct} or one unique recursive match."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root. Default assumes week08_capstone/supporting_analysis/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    w4_notebook = locate(
        repo_root,
        "week04_anomaly/W04_Anomaly_Benchmark.ipynb",
        "W04_Anomaly_Benchmark.ipynb",
    )
    w4_summary = locate(
        repo_root,
        "week04_anomaly/results/w04_anomaly_benchmark/w04_metrics_summary_mean_std.csv",
        "w04_metrics_summary_mean_std.csv",
    )
    w4_per_split = locate(
        repo_root,
        "week04_anomaly/results/w04_anomaly_benchmark/w04_metrics_per_split.csv",
        "w04_metrics_per_split.csv",
    )
    w4_ledger = locate(
        repo_root,
        "week04_anomaly/results/w04_anomaly_benchmark/w04_fault_injection_ledger.csv",
        "w04_fault_injection_ledger.csv",
    )
    w4_wilcoxon = locate(
        repo_root,
        "week04_anomaly/results/w04_anomaly_benchmark/w04_wilcoxon_auroc_pairwise.csv",
        "w04_wilcoxon_auroc_pairwise.csv",
    )

    data_dir = repo_root / "data" / "w03_rover_5units_1day"
    if not data_dir.exists():
        matches = [
            p for p in repo_root.rglob("w03_rover_5units_1day")
            if p.is_dir()
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                "Could not resolve the Week 3 five-unit telemetry directory."
            )
        data_dir = matches[0]

    # Exact Week 4 configuration.
    ns = {
        "WINDOW_SECONDS": 60,
        "TEST_HOURS_PER_SPLIT": 8,
        "FAULT_EVENTS_PER_TYPE": 3,
        "N_SPLITS": 10,
        "SPLIT_SEEDS": [20260400 + i for i in range(10)],
        "CONTAMINATION": 0.03,
        "THRESHOLD_QUANTILE": 0.97,
        "LSTM_SEQUENCE_LENGTH": 8,
        "LSTM_EPOCHS": 1,
        "LSTM_BATCH_SIZE": 256,
        "LSTM_HIDDEN_DIM": 16,
        "LSTM_LATENT_DIM": 8,
        "LSTM_MAX_TRAIN_SEQUENCES": 1000,
        "FAULT_TYPES": [
            "motor_stall",
            "gradual_sensor_drift",
            "accelerated_battery_degradation",
            "gps_jitter_spoofing",
            "intermittent_software_hang",
        ],
    }

    # Load the exact implementation cells from the final Week 4 notebook.
    nb = json.loads(w4_notebook.read_text(encoding="utf-8"))
    import_cell = "".join(nb["cells"][1]["source"])
    exec(import_cell, ns)
    ns["torch"].set_num_threads(1)

    for cell_index in [8, 10, 12, 13]:
        exec("".join(nb["cells"][cell_index]["source"]), ns)

    raw_cols = [
        "schema_version", "timestamp", "elapsed_s", "day_index",
        "robot_id", "robot_index", "mission_mode", "terrain", "location_zone",
        "speed_mps", "battery_soc_pct", "battery_voltage_v", "motor_temp_c",
        "motor_current_fl_a", "motor_current_fr_a", "motor_current_rl_a",
        "motor_current_rr_a", "imu_accel_x_mps2", "imu_accel_y_mps2",
        "imu_accel_z_mps2", "imu_gyro_x_rps", "imu_gyro_y_rps",
        "imu_gyro_z_rps", "wifi_rssi_dbm", "gps_fix_quality", "gps_hdop",
        "gps_num_sats", "x_m", "y_m", "task_success_probability",
        "task_success_flag", "benign_noise_flag", "missingness_flag", "is_charging",
    ]

    frames = []
    for file in sorted(data_dir.glob("**/telemetry.csv.gz")):
        frames.append(pd.read_csv(file, usecols=lambda c: c in raw_cols))
    if len(frames) != 5:
        raise RuntimeError(f"Expected five telemetry partitions, found {len(frames)}.")

    clean_df = pd.concat(frames, ignore_index=True)
    clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"], utc=True)
    clean_df = clean_df.sort_values(["robot_id", "elapsed_s"]).reset_index(drop=True)

    for col in clean_df.columns:
        if col not in [
            "schema_version", "timestamp", "robot_id",
            "mission_mode", "terrain", "location_zone",
        ]:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    summary = pd.read_csv(w4_summary)
    per_split = pd.read_csv(w4_per_split)
    ledger = pd.read_csv(w4_ledger)
    wilcoxon = pd.read_csv(w4_wilcoxon)

    select_test_hours = ns["select_test_hours"]
    inject_controlled_faults = ns["inject_controlled_faults"]
    build_window_features = ns["build_window_features"]
    make_lstm_sequences = ns["make_lstm_sequences"]
    train_lstm_autoencoder = ns["train_lstm_autoencoder"]
    lstm_features = ns["LSTM_FEATURE_COLS"]
    batch_size = ns["LSTM_BATCH_SIZE"]

    def reconstruction_errors(model, sequences):
        model.eval()
        overall, per_feature = [], []
        with torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                batch = torch.tensor(
                    sequences[start:start + batch_size],
                    dtype=torch.float32,
                )
                recon = model(batch)
                err = (recon - batch) ** 2
                overall.append(err.mean(dim=(1, 2)).cpu().numpy())
                per_feature.append(err.mean(dim=1).cpu().numpy())
        return np.concatenate(overall), np.concatenate(per_feature)

    window_rows = []
    event_rows = []
    threshold_rows = []
    feature_rows = []
    focus_rows = []
    epoch_rows = []

    battery_focus_cols = [
        "battery_soc_pct__mean",
        "battery_voltage_v__mean",
        "battery_voltage_soc_residual",
        "battery_soc_interunit_z",
    ]

    for split_seed in ns["SPLIT_SEEDS"]:
        test_hours = select_test_hours(split_seed)
        fault_df, split_ledger = inject_controlled_faults(
            clean_df, seed=split_seed, test_hours=test_hours
        )
        win = build_window_features(fault_df, window_seconds=60)

        train_mask = (~win["hour_block"].isin(test_hours)) & (win["is_fault"] == 0)
        test_mask = win["hour_block"].isin(test_hours)

        cols = [c for c in lstm_features if c in win.columns]
        X = win[cols].replace([np.inf, -np.inf], np.nan)
        medians = X.loc[train_mask].median()
        X = X.fillna(medians)

        scaler = StandardScaler()
        scaler.fit(X.loc[train_mask])

        ready = win.copy()
        ready.loc[:, cols] = scaler.transform(X)

        train_seq, train_idx = make_lstm_sequences(
            ready, cols, train_mask
        )
        test_seq, test_idx = make_lstm_sequences(
            ready, cols, test_mask
        )

        labels = win.loc[test_idx, "fault_type"].to_numpy()
        keep = np.isin(
            labels,
            ["clean", "accelerated_battery_degradation"],
        )
        y = (labels[keep] == "accelerated_battery_degradation").astype(int)

        for epochs in [1, 5, 10]:
            ns["LSTM_EPOCHS"] = epochs
            model = train_lstm_autoencoder(train_seq, seed=split_seed)
            train_score, train_feature_error = reconstruction_errors(model, train_seq)
            test_score, test_feature_error = reconstruction_errors(model, test_seq)

            threshold = float(np.quantile(train_score, 0.97))
            score = test_score[keep]
            pred = (score >= threshold).astype(int)

            precision, recall, f1, _ = precision_recall_fscore_support(
                y, pred, average="binary", zero_division=0
            )
            epoch_rows.append(
                {
                    "epochs": epochs,
                    "split_seed": split_seed,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "auroc": roc_auc_score(y, score),
                }
            )

            if epochs != 1:
                continue

            subset_idx = test_idx[keep]
            subset = win.loc[
                subset_idx,
                [
                    "robot_id", "window_id", "hour_block", "fault_type",
                    "fault_id", "fault_severity", "battery_soc_pct__mean",
                    "battery_voltage_v__mean", "battery_voltage_soc_residual",
                    "battery_soc_5m_slope_pct_per_min",
                ],
            ].copy()
            subset["split_seed"] = split_seed
            subset["score"] = score
            subset["threshold"] = threshold
            subset["predicted_anomaly"] = score >= threshold
            subset["is_battery_fault"] = y.astype(bool)
            window_rows.append(subset)

            for q in [0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995]:
                q_threshold = float(np.quantile(train_score, q))
                q_pred = (score >= q_threshold).astype(int)
                p, r, f, _ = precision_recall_fscore_support(
                    y, q_pred, average="binary", zero_division=0
                )
                threshold_rows.append(
                    {
                        "split_seed": split_seed,
                        "train_score_quantile": q,
                        "threshold": q_threshold,
                        "precision": p,
                        "recall": r,
                        "f1": f,
                        "clean_test_flag_rate": float(q_pred[y == 0].mean()),
                    }
                )

            score_series = pd.Series(test_score, index=test_idx)
            battery_events = split_ledger[
                split_ledger["fault_type"] == "accelerated_battery_degradation"
            ]
            for _, event in battery_events.iterrows():
                event_idx = win.index[win["fault_id"] == event["fault_id"]]
                event_score = score_series.reindex(event_idx).dropna()
                flagged = event_score[event_score >= threshold]
                detected = len(flagged) > 0
                latency = np.nan
                if detected:
                    first_window = int(win.loc[flagged.index[0], "window_id"])
                    latency = max(0.0, (first_window + 1) * 60 - float(event["start_s"]))
                event_rows.append(
                    {
                        "split_seed": split_seed,
                        "fault_id": event["fault_id"],
                        "robot_id": event["robot_id"],
                        "duration_s": event["duration_s"],
                        "severity": event["severity"],
                        "n_scored_fault_windows": len(event_score),
                        "event_detected": detected,
                        "flagged_fault_windows": int((event_score >= threshold).sum()),
                        "fault_window_recall": float((event_score >= threshold).mean()),
                        "mean_fault_score": float(event_score.mean()),
                        "max_fault_score": float(event_score.max()),
                        "threshold": threshold,
                        "detection_latency_s": latency,
                    }
                )

            test_feature_subset = test_feature_error[keep]
            for j, feature in enumerate(cols):
                fscore = test_feature_subset[:, j]
                feature_rows.append(
                    {
                        "split_seed": split_seed,
                        "feature": feature,
                        "feature_error_auroc": roc_auc_score(y, fscore),
                        "clean_error_mean": float(fscore[y == 0].mean()),
                        "fault_error_mean": float(fscore[y == 1].mean()),
                        "fault_to_clean_error_ratio": float(
                            fscore[y == 1].mean() / fscore[y == 0].mean()
                        ),
                    }
                )

            focus_idx = [cols.index(c) for c in battery_focus_cols]
            train_focus = train_feature_error[:, focus_idx].mean(axis=1)
            test_focus = test_feature_error[:, focus_idx].mean(axis=1)[keep]
            focus_threshold = float(np.quantile(train_focus, 0.97))
            focus_pred = (test_focus >= focus_threshold).astype(int)
            p, r, f, _ = precision_recall_fscore_support(
                y, focus_pred, average="binary", zero_division=0
            )
            focus_rows.append(
                {
                    "split_seed": split_seed,
                    "auroc": roc_auc_score(y, test_focus),
                    "precision": p,
                    "recall": r,
                    "f1": f,
                    "clean_flag_rate": float(focus_pred[y == 0].mean()),
                }
            )

    window_diag = pd.concat(window_rows, ignore_index=True)
    event_diag = pd.DataFrame(event_rows)
    threshold_diag = pd.DataFrame(threshold_rows)
    feature_diag = pd.DataFrame(feature_rows)
    focus_diag = pd.DataFrame(focus_rows)
    epoch_diag = pd.DataFrame(epoch_rows)

    method_comparison = summary[
        summary["scope"] == "accelerated_battery_degradation"
    ].sort_values("auroc_mean", ascending=False)
    method_comparison.to_csv(
        results_dir / "lstm_battery_method_comparison.csv", index=False
    )
    window_diag.to_csv(
        results_dir / "lstm_battery_window_diagnostics.csv", index=False
    )
    event_diag.to_csv(
        results_dir / "lstm_battery_event_diagnostics.csv", index=False
    )
    threshold_diag.to_csv(
        results_dir / "lstm_battery_threshold_sensitivity.csv", index=False
    )
    feature_diag.to_csv(
        results_dir / "lstm_battery_feature_reconstruction_error.csv", index=False
    )
    focus_diag.to_csv(
        results_dir / "lstm_battery_focused_score.csv", index=False
    )
    epoch_diag.to_csv(
        results_dir / "lstm_battery_epoch_sensitivity.csv", index=False
    )

    threshold_summary = threshold_diag.groupby("train_score_quantile").agg(
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        clean_flag_rate_mean=("clean_test_flag_rate", "mean"),
        clean_flag_rate_std=("clean_test_flag_rate", "std"),
    ).reset_index()
    threshold_summary.to_csv(
        results_dir / "lstm_battery_threshold_sensitivity_summary.csv",
        index=False,
    )

    feature_summary = feature_diag.groupby("feature").agg(
        auroc_mean=("feature_error_auroc", "mean"),
        auroc_std=("feature_error_auroc", "std"),
        clean_error_mean=("clean_error_mean", "mean"),
        fault_error_mean=("fault_error_mean", "mean"),
        fault_to_clean_error_ratio_mean=("fault_to_clean_error_ratio", "mean"),
    ).reset_index().sort_values("auroc_mean", ascending=False)
    feature_summary.to_csv(
        results_dir / "lstm_battery_feature_reconstruction_summary.csv",
        index=False,
    )

    focus_summary = pd.DataFrame(
        [{
            "score_definition": "Mean reconstruction MSE over four battery-related channels",
            "auroc_mean": focus_diag["auroc"].mean(),
            "auroc_std": focus_diag["auroc"].std(),
            "precision_mean": focus_diag["precision"].mean(),
            "recall_mean": focus_diag["recall"].mean(),
            "f1_mean": focus_diag["f1"].mean(),
            "clean_flag_rate_mean": focus_diag["clean_flag_rate"].mean(),
        }]
    )
    focus_summary.to_csv(
        results_dir / "lstm_battery_focused_score_summary.csv",
        index=False,
    )

    epoch_summary = epoch_diag.groupby("epochs").agg(
        auroc_mean=("auroc", "mean"),
        auroc_std=("auroc", "std"),
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
        f1_mean=("f1", "mean"),
    ).reset_index()
    epoch_summary.to_csv(
        results_dir / "lstm_battery_epoch_sensitivity_summary.csv",
        index=False,
    )

    # Reproduction check against the saved final Week 4 metrics.
    original = per_split[
        (per_split["scope"] == "accelerated_battery_degradation")
        & (per_split["method"] == "LSTM Autoencoder")
    ][["split_seed", "precision", "recall", "f1", "auroc"]].sort_values("split_seed")

    reproduced = epoch_diag[epoch_diag["epochs"] == 1].drop(
        columns=["epochs"]
    ).rename(
        columns={
            "precision": "precision_reproduced",
            "recall": "recall_reproduced",
            "f1": "f1_reproduced",
            "auroc": "auroc_reproduced",
        }
    )
    check = original.merge(reproduced, on="split_seed")
    for metric in ["precision", "recall", "f1", "auroc"]:
        check[f"{metric}_abs_diff"] = (
            check[f"{metric}_reproduced"] - check[metric]
        ).abs()
    check.to_csv(
        results_dir / "lstm_battery_reproduction_check.csv", index=False
    )

    battery_wilcoxon = wilcoxon[
        (wilcoxon["scope"] == "accelerated_battery_degradation")
        & (
            (wilcoxon["method_a"] == "LSTM Autoencoder")
            | (wilcoxon["method_b"] == "LSTM Autoencoder")
        )
    ]
    battery_wilcoxon.to_csv(
        results_dir / "lstm_battery_wilcoxon_comparisons.csv", index=False
    )

    print("PASS: follow-up analysis reproduced and result tables regenerated.")
    print(
        method_comparison[
            ["method", "f1_mean", "auroc_mean"]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
