#!/usr/bin/env python3
"""Audit completeness, parsing, reliability, and judge-ground-truth alignment."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import PROJECT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "results" / "llm_judge_all_runs.csv"),
    )
    parser.add_argument("--expected-scenarios", type=int, default=50)
    parser.add_argument("--expected-seeds", type=int, nargs="*", default=[7, 17, 27])
    args = parser.parse_args()

    path = Path(args.input)
    frame = pd.read_csv(path)
    frame["parse_ok"] = frame["parse_ok"].astype(bool)
    valid = frame[frame["parse_ok"]].copy()

    print("=== Completeness ===")
    print("Rows:", len(frame))
    print("Unique scenarios:", frame["scenario_id"].nunique())
    print("Judge seeds:", sorted(frame["judge_seed"].unique().tolist()))
    print(frame.groupby("judge_seed")["scenario_id"].agg(["count", "nunique"]).to_string())

    print("\n=== Parsing ===")
    print(f"Parse rate: {frame['parse_ok'].mean():.3f}")
    print(f"Retry rate: {frame['used_retry'].astype(bool).mean():.3f}")
    print("Invalid outputs:", int((~frame["parse_ok"]).sum()))
    if (~frame["parse_ok"]).any():
        print(
            frame.loc[
                ~frame["parse_ok"],
                ["scenario_id", "judge_seed", "raw_text"],
            ].to_string(index=False)
        )

    print("\n=== Verdict distribution among valid judgments ===")
    print(valid["verdict"].value_counts(dropna=False).to_string())
    print("\nBy seed:")
    print(pd.crosstab(valid["judge_seed"], valid["verdict"]).to_string())
    print("\nBy domain:")
    print(pd.crosstab(valid["domain"], valid["verdict"]).to_string())

    print("\n=== Exact match versus verdict ===")
    print(pd.crosstab(valid["exact_match"], valid["verdict"], margins=True).to_string())
    if len(valid):
        exact = valid[valid["exact_match"].astype(bool)]
        mismatch = valid[~valid["exact_match"].astype(bool)]
        print(
            "PASS rate when exact_match=True:",
            round((exact["verdict"] == "PASS").mean(), 3) if len(exact) else "n/a",
        )
        print(
            "PASS rate when exact_match=False:",
            round((mismatch["verdict"] == "PASS").mean(), 3) if len(mismatch) else "n/a",
        )

    print("\n=== Seed disagreement ===")
    pivot = valid.pivot(index="scenario_id", columns="judge_seed", values="verdict")
    complete = pivot.dropna()
    disagree = complete[complete.nunique(axis=1) > 1]
    print("Complete three-run scenarios:", len(complete))
    print("Disagreement scenarios:", len(disagree))
    if len(disagree):
        print(disagree.to_string())

    reliability_path = PROJECT_ROOT / "results" / "reliability_summary.csv"
    kappa_path = PROJECT_ROOT / "results" / "pairwise_kappa.csv"
    if reliability_path.exists():
        print("\n=== Krippendorff alpha ===")
        print(pd.read_csv(reliability_path).to_string(index=False))
    if kappa_path.exists():
        print("\n=== Pairwise Cohen kappa ===")
        print(pd.read_csv(kappa_path).to_string(index=False))

    expected_scenarios = int(args.expected_scenarios)
    expected_seeds = set(args.expected_seeds)
    expected_rows = expected_scenarios * len(expected_seeds)
    print("\n=== Basic assertions ===")
    assert len(frame) == expected_rows, f"Expected {expected_rows} rows, found {len(frame)}"
    assert frame["scenario_id"].nunique() == expected_scenarios
    assert set(frame["judge_seed"]) == expected_seeds
    assert not frame.duplicated(["scenario_id", "judge_seed"]).any()
    print("PASS: 50 scenarios, 150 unique seeded judgments, and no duplicate pairs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
