"""Shared utilities for the Week 5 reinforcement-learning benchmarks."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "w05_rl_config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the YAML experiment configuration."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_output_dirs() -> None:
    """Create all expected artifact directories."""
    for relative in [
        "results/raw/single_agent",
        "results/raw/multi_agent",
        "plots",
        "tensorboard",
        "models",
    ]:
        (PROJECT_ROOT / relative).mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    """Convert a label to a filesystem-safe slug."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def package_version(name: str) -> str:
    """Return an installed package version or 'not-installed'."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def system_metadata() -> dict[str, Any]:
    """Capture software and machine metadata for reproducibility."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "gymnasium": package_version("gymnasium"),
        "stable_baselines3": package_version("stable-baselines3"),
        "torch": package_version("torch"),
        "pettingzoo": package_version("pettingzoo"),
        "mpe2": package_version("mpe2"),
        "supersuit": package_version("supersuit"),
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
    }


def json_default(value: Any) -> Any:
    """JSON serializer for numpy and Path objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write an indented JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )


def make_single_agent_env(
    env_key: str,
    env_config: dict[str, Any],
    seed: int,
    monitor_path: Path | None = None,
) -> tuple[gym.Env, str]:
    """
    Create a seeded Gymnasium environment.

    The work plan names LunarLanderContinuous-v3. Current Gymnasium exposes
    continuous control through LunarLander-v3 with continuous=True, so this
    helper tries the requested ID first and records the fallback when needed.
    """
    requested_id = env_config["requested_id"]
    fallback_id = env_config.get("fallback_id")
    fallback_kwargs = env_config.get("fallback_kwargs", {})

    try:
        env = gym.make(requested_id)
        actual_id = requested_id
    except Exception as requested_error:
        if fallback_id is None:
            raise RuntimeError(
                f"Could not create requested environment {requested_id!r}."
            ) from requested_error
        env = gym.make(fallback_id, **fallback_kwargs)
        actual_id = f"{fallback_id}({fallback_kwargs})"

    set_random_seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    env.reset(seed=seed)

    if monitor_path is not None:
        monitor_path.parent.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, filename=str(monitor_path))

    return env, actual_id


def normalized_curve_auc(timesteps: np.ndarray, rewards: np.ndarray) -> float:
    """Compute reward-versus-timestep area normalized by final timestep."""
    if len(timesteps) < 2 or float(timesteps[-1]) <= 0:
        return float("nan")
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return float(integrate(rewards, timesteps) / float(timesteps[-1]))


def reward_at_fraction(
    timesteps: np.ndarray,
    rewards: np.ndarray,
    fraction: float,
    total_timesteps: int,
) -> float:
    """Linearly interpolate evaluation reward at a fraction of the budget."""
    if len(timesteps) == 0:
        return float("nan")
    target = float(total_timesteps) * fraction
    return float(np.interp(target, timesteps, rewards))
