#!/usr/bin/env python3
"""Run the full Week 5 benchmark grid, then aggregate and plot results."""

from __future__ import annotations

import argparse
import subprocess
import sys

from common import DEFAULT_CONFIG, PROJECT_ROOT, load_config


def run_command(command: list[str], continue_on_error: bool) -> bool:
    print("\n$", " ".join(command))
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if result.returncode != 0:
        print(f"FAILED with exit code {result.returncode}")
        if not continue_on_error:
            raise SystemExit(result.returncode)
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--profile", default="standard", choices=["smoke", "standard", "extended"])
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--include", choices=["single", "multi"], nargs="+", default=["single", "multi"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    seeds = args.seeds or list(config["project"]["seeds"])
    python = sys.executable
    common_flags = ["--config", str(args.config), "--profile", args.profile]
    if args.device:
        common_flags += ["--device", args.device]
    if args.overwrite:
        common_flags.append("--overwrite")
    if args.progress:
        common_flags.append("--progress")

    completed = 0
    failed = 0

    if "single" in args.include:
        for environment in config["single_agent"]["environments"]:
            for algorithm in config["single_agent"]["algorithms"]:
                for seed in seeds:
                    command = [
                        python,
                        str(PROJECT_ROOT / "scripts" / "train_single_agent.py"),
                        "--environment",
                        environment,
                        "--algorithm",
                        algorithm,
                        "--seed",
                        str(seed),
                        *common_flags,
                    ]
                    ok = run_command(command, args.continue_on_error)
                    completed += int(ok)
                    failed += int(not ok)

    if "multi" in args.include:
        for seed in seeds:
            command = [
                python,
                str(PROJECT_ROOT / "scripts" / "train_multi_agent.py"),
                "--seed",
                str(seed),
                *common_flags,
            ]
            ok = run_command(command, args.continue_on_error)
            completed += int(ok)
            failed += int(not ok)

    run_command(
        [python, str(PROJECT_ROOT / "scripts" / "aggregate_results.py")],
        continue_on_error=False,
    )
    run_command(
        [python, str(PROJECT_ROOT / "scripts" / "plot_learning_curves.py")],
        continue_on_error=False,
    )

    print(f"\nFinished. Successful commands: {completed}; failed commands: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
