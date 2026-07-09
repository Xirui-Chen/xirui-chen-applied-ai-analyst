#!/usr/bin/env python3
"""
W03_Telemetry_Generator.py

Synthetic Aido Rover fleet telemetry generator.

This script creates deterministic, partitioned 1 Hz telemetry for a synthetic
Aido Rover patrol fleet. The default run supports 50 simulated Rover units over
30 days, but the output is written by robot-day partition so the full dataset
does not need to fit in memory.

Generated signals include:
    * joint and actuator angles
    * IMU acceleration and gyro, 3 axes each
    * per-drive motor currents
    * battery voltage and state of charge
    * WiFi RSSI
    * GPS fix quality
    * discrete task-success flags
    * contextual fields for terrain, patrol mode, weather, location zone,
      benign measurement noise, and light missingness

Example smoke test:
    python W03_Telemetry_Generator.py \
        --seed 42 \
        --fleet-size 2 \
        --horizon-days 0.02 \
        --output-dir data/w03_rover_smoke \
        --format csv.gz \
        --verify-reproducibility

Default full generation:
    python W03_Telemetry_Generator.py \
        --seed 42 \
        --fleet-size 50 \
        --horizon-days 30 \
        --output-dir data/w03_rover_telemetry \
        --format csv.gz

Dry-run estimate:
    python W03_Telemetry_Generator.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


SCHEMA_VERSION = "w03_rover_telemetry_v1"
SECONDS_PER_DAY = 86_400
DEFAULT_START_TS = "2026-07-01T00:00:00Z"

TERRAIN_NAMES = np.array(["asphalt", "gravel", "dry_grass", "wet_grass", "mud", "sand", "rocky"], dtype=object)
TERRAIN_FRICTION = np.array([0.90, 0.60, 0.50, 0.30, 0.20, 0.15, 0.70], dtype=float)
TERRAIN_VIBRATION = np.array([0.04, 0.10, 0.12, 0.18, 0.24, 0.28, 0.16], dtype=float)
TERRAIN_SPEED_FACTOR = np.array([1.00, 0.88, 0.82, 0.70, 0.55, 0.50, 0.78], dtype=float)

ZONE_NAMES = np.array(["base_area", "open_perimeter", "tree_line", "building_shadow", "remote_edge"], dtype=object)
ZONE_RSSI_BASE = np.array([-48.0, -58.0, -66.0, -73.0, -79.0], dtype=float)
ZONE_MULTIPATH_AMP = np.array([1.5, 3.0, 5.5, 8.0, 10.0], dtype=float)
ZONE_GPS_PENALTY = np.array([0.0, 0.3, 0.8, 1.3, 1.6], dtype=float)

MODE_BASE_SPEED = {
    "coverage": 1.00,
    "perimeter": 1.00,
    "incident_response": 1.50,
    "escort": 0.70,
    "pursuit": 1.60,
    "return_to_dock": 0.90,
    "charging": 0.00,
}


@dataclass(frozen=True)
class GeneratorConfig:
    """Configuration for the synthetic telemetry generator."""

    seed: int = 42
    fleet_size: int = 50
    horizon_days: float = 30.0
    freq_hz: float = 1.0
    start_ts: str = DEFAULT_START_TS
    output_dir: str = "data/w03_rover_telemetry"
    output_format: str = "csv.gz"
    partition: str = "robot_day"
    benign_noise_rate: float = 0.015
    missing_rate: float = 0.004
    float_format: str = "%.6f"
    force: bool = False
    write_manifest: bool = True

    @property
    def total_ticks(self) -> int:
        return int(round(self.horizon_days * SECONDS_PER_DAY * self.freq_hz))

    @property
    def ticks_per_day(self) -> int:
        return int(round(SECONDS_PER_DAY * self.freq_hz))

    @property
    def dt(self) -> float:
        return 1.0 / self.freq_hz


def stable_int_seed(*parts: object) -> int:
    """Return a stable 64-bit integer seed from arbitrary input parts."""
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def rng_for(config: GeneratorConfig, robot_index: int, partition_index: int, stream: str) -> np.random.Generator:
    """Return an independent deterministic RNG for one stream."""
    return np.random.default_rng(stable_int_seed(config.seed, robot_index, partition_index, stream))


def robot_id(robot_index: int) -> str:
    return f"AR-{robot_index + 1:03d}"


def validate_config(config: GeneratorConfig) -> None:
    if config.fleet_size <= 0:
        raise ValueError("--fleet-size must be positive")
    if config.horizon_days <= 0:
        raise ValueError("--horizon-days must be positive")
    if config.freq_hz <= 0:
        raise ValueError("--freq-hz must be positive")
    if config.output_format not in {"csv", "csv.gz", "parquet"}:
        raise ValueError("--format must be one of: csv, csv.gz, parquet")
    if config.partition != "robot_day":
        raise ValueError("Only --partition robot_day is currently supported")
    if not (0.0 <= config.benign_noise_rate <= 0.10):
        raise ValueError("--benign-noise-rate must be between 0 and 0.10")
    if not (0.0 <= config.missing_rate <= 0.10):
        raise ValueError("--missing-rate must be between 0 and 0.10")


def partition_ranges(config: GeneratorConfig) -> Iterable[Tuple[int, int, int]]:
    """Yield (day_index, start_tick, n_ticks) partitions."""
    start_tick = 0
    day_index = 0
    while start_tick < config.total_ticks:
        n_ticks = min(config.ticks_per_day, config.total_ticks - start_tick)
        yield day_index, start_tick, n_ticks
        start_tick += n_ticks
        day_index += 1


def build_timestamp_series(config: GeneratorConfig, start_tick: int, n_ticks: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(config.start_ts)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    ticks = np.arange(start_tick, start_tick + n_ticks, dtype=np.int64)
    seconds = ticks / config.freq_hz
    return start + pd.to_timedelta(seconds, unit="s")


def battery_profile(config: GeneratorConfig, robot_index: int, day_index: int, elapsed_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return battery SoC, charging flag, and cycle phase seconds."""
    cycle_s = 10.0 * 3600.0
    discharge_s = 8.0 * 3600.0
    charge_s = cycle_s - discharge_s
    phase_offset_s = (robot_index * 731.0 + (config.seed % 997)) % cycle_s
    phase_s = (elapsed_s + phase_offset_s) % cycle_s
    is_charging = phase_s >= discharge_s
    discharge_soc = 95.0 - 85.0 * (phase_s / discharge_s)
    charge_soc = 10.0 + 85.0 * ((phase_s - discharge_s) / charge_s)
    soc = np.where(is_charging, charge_soc, discharge_soc)
    soc = np.clip(soc - 0.012 * day_index, 5.0, 98.0)
    rng = rng_for(config, robot_index, day_index, "battery")
    soc = np.clip(soc + rng.normal(0.0, 0.12, size=elapsed_s.size), 4.0, 100.0)
    return soc, is_charging, phase_s


def terrain_and_zone(robot_index: int, day_index: int, elapsed_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return terrain and location-zone context arrays."""
    terrain_idx = (
        np.floor((elapsed_s + robot_index * 43.0 + day_index * 19.0) / (12 * 60)).astype(int)
        + robot_index
        + day_index
    ) % len(TERRAIN_NAMES)
    zone_idx = (
        np.floor((elapsed_s + robot_index * 113.0 + day_index * 29.0) / (15 * 60)).astype(int)
        + 2 * robot_index
        + day_index
    ) % len(ZONE_NAMES)
    return terrain_idx, TERRAIN_NAMES[terrain_idx], TERRAIN_FRICTION[terrain_idx], ZONE_NAMES[zone_idx], zone_idx


def weather(config: GeneratorConfig, robot_index: int, day_index: int, seconds_of_day: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return ambient temperature and rain intensity."""
    rng = rng_for(config, robot_index, day_index, "weather")
    day_phase = 2.0 * np.pi * seconds_of_day / SECONDS_PER_DAY
    ambient_c = 22.0 + 8.0 * np.sin(day_phase - np.pi / 3.0) + rng.normal(0.0, 0.3, size=seconds_of_day.size)
    ambient_c += rng.normal(0.0, 0.8)
    if rng.random() < 0.24:
        center = rng.uniform(0.25 * SECONDS_PER_DAY, 0.80 * SECONDS_PER_DAY)
        width = rng.uniform(2_000.0, 7_000.0)
        rain = np.exp(-0.5 * ((seconds_of_day - center) / width) ** 2) * rng.uniform(0.3, 1.0)
    else:
        rain = np.zeros_like(seconds_of_day, dtype=float)
    return ambient_c, rain


def mode_schedule(config: GeneratorConfig, robot_index: int, day_index: int, elapsed_s: np.ndarray, is_charging: np.ndarray, soc_pct: np.ndarray) -> np.ndarray:
    """Return deterministic patrol modes."""
    rng = rng_for(config, robot_index, day_index, "modes")
    segment_ids = np.floor((elapsed_s + robot_index * 137.0) / (30 * 60)).astype(np.int64)
    unique_segments = np.unique(segment_ids)
    choices = np.array(["coverage", "perimeter", "incident_response", "escort", "pursuit"], dtype=object)
    probs = np.array([0.40, 0.32, 0.10, 0.10, 0.08])
    segment_modes = rng.choice(choices, size=unique_segments.size, p=probs)
    mapping = dict(zip(unique_segments.tolist(), segment_modes.tolist()))
    modes = np.array([mapping[int(seg)] for seg in segment_ids], dtype=object)
    modes = np.where(soc_pct <= 12.0, "return_to_dock", modes)
    modes = np.where(is_charging, "charging", modes)
    return modes


def speed_profile(config: GeneratorConfig, robot_index: int, day_index: int, modes: np.ndarray, terrain_idx: np.ndarray, rain: np.ndarray, elapsed_s: np.ndarray) -> np.ndarray:
    """Return speed in metres per second."""
    rng = rng_for(config, robot_index, day_index, "speed")
    base = np.array([MODE_BASE_SPEED[str(m)] for m in modes], dtype=float)
    speed = base * TERRAIN_SPEED_FACTOR[terrain_idx] * (1.0 - 0.22 * rain)
    speed += 0.05 * np.sin(2.0 * np.pi * elapsed_s / 120.0 + robot_index)
    speed += rng.normal(0.0, 0.035, size=elapsed_s.size)
    speed = np.where(modes == "charging", 0.0, speed)
    return np.clip(speed, 0.0, 2.05)


def motor_current_and_temp(config: GeneratorConfig, robot_index: int, day_index: int, speed: np.ndarray, friction: np.ndarray, rain: np.ndarray, ambient_c: np.ndarray, elapsed_s: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Return per-drive motor currents and motor temperature."""
    rng = rng_for(config, robot_index, day_index, "motor")
    terrain_load = (1.0 / np.maximum(friction, 0.05) - 1.0) * 2.8
    motor_temp_c = ambient_c + 9.0 + 7.5 * speed + 2.0 * terrain_load
    motor_temp_c += 1.8 * np.sin(2.0 * np.pi * elapsed_s / 3600.0 + robot_index)
    motor_temp_c += rng.normal(0.0, 0.5, size=speed.size)
    thermal_drift_a = np.maximum(motor_temp_c - 40.0, 0.0) * 0.060
    common = 5.0 + 8.8 * speed + terrain_load * speed
    common = common * (1.0 + 0.12 * rain) * rng.normal(1.0, 0.04) + thermal_drift_a
    common += rng.normal(0.0, 0.25, size=speed.size)
    turn = 0.10 * np.sin(2.0 * np.pi * elapsed_s / 75.0 + 0.31 * robot_index)
    front = 1.0 + 0.025 * np.sin(2.0 * np.pi * elapsed_s / 211.0)
    rear = 1.0 - 0.015 * np.sin(2.0 * np.pi * elapsed_s / 173.0)
    currents = {
        "motor_current_fl_a": np.clip(common * (1.0 + turn) * front + rng.normal(0.0, 0.18, size=speed.size), 0.0, 60.0),
        "motor_current_fr_a": np.clip(common * (1.0 - turn) * front + rng.normal(0.0, 0.18, size=speed.size), 0.0, 60.0),
        "motor_current_rl_a": np.clip(common * (1.0 + turn) * rear + rng.normal(0.0, 0.18, size=speed.size), 0.0, 60.0),
        "motor_current_rr_a": np.clip(common * (1.0 - turn) * rear + rng.normal(0.0, 0.18, size=speed.size), 0.0, 60.0),
    }
    return currents, motor_temp_c


def battery_voltage(soc: np.ndarray, is_charging: np.ndarray, currents: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return 48 V nominal battery voltage with load sag."""
    total_current = sum(currents[col] for col in currents)
    open_circuit_v = 40.0 + 11.2 * (soc / 100.0)
    return np.clip(open_circuit_v - 0.0065 * total_current + np.where(is_charging, 1.4, 0.0), 39.0, 53.2)


def imu_signals(config: GeneratorConfig, robot_index: int, day_index: int, speed: np.ndarray, terrain_idx: np.ndarray, rain: np.ndarray, elapsed_s: np.ndarray) -> Dict[str, np.ndarray]:
    """Return IMU accelerometer and gyro signals."""
    rng = rng_for(config, robot_index, day_index, "imu")
    vibration = TERRAIN_VIBRATION[terrain_idx] * (0.5 + speed) + 0.04 * rain
    turn_rate = 0.060 * np.sin(2.0 * np.pi * elapsed_s / 65.0 + robot_index * 0.2)
    turn_rate += 0.025 * np.sin(2.0 * np.pi * elapsed_s / 17.0)
    return {
        "imu_accel_x_mps2": np.gradient(speed, config.dt) + rng.normal(0.0, 0.035, size=speed.size) + 0.3 * vibration,
        "imu_accel_y_mps2": speed * turn_rate + rng.normal(0.0, 0.025, size=speed.size),
        "imu_accel_z_mps2": 9.81 + rng.normal(0.0, 0.035, size=speed.size) + vibration * rng.normal(0.0, 0.55, size=speed.size),
        "imu_gyro_x_rps": rng.normal(0.0, 0.010, size=speed.size) + 0.02 * vibration,
        "imu_gyro_y_rps": rng.normal(0.0, 0.010, size=speed.size) - 0.015 * vibration,
        "imu_gyro_z_rps": turn_rate + rng.normal(0.0, 0.012, size=speed.size),
    }


def joint_angles(config: GeneratorConfig, robot_index: int, day_index: int, speed: np.ndarray, elapsed_s: np.ndarray) -> Dict[str, np.ndarray]:
    """Return wheel, steering, mast, and suspension actuator proxies."""
    rng = rng_for(config, robot_index, day_index, "joints")
    wheel_radius_m = 0.165
    base_angle = np.mod(np.cumsum(speed * config.dt) / wheel_radius_m + robot_index * 0.13, 2.0 * np.pi)
    turn_adjust = 0.050 * np.sin(2.0 * np.pi * elapsed_s / 75.0 + robot_index)
    steer = 0.22 * np.sin(2.0 * np.pi * elapsed_s / 180.0 + robot_index / 5.0)
    sensor_pan = np.mod(0.020 * elapsed_s + 0.3 * np.sin(2.0 * np.pi * elapsed_s / 600.0), 2.0 * np.pi)
    suspension = 0.06 * np.sin(2.0 * np.pi * elapsed_s / 4.5) * np.clip(speed, 0.0, 1.5)
    suspension += rng.normal(0.0, 0.005, size=speed.size)
    return {
        "wheel_fl_angle_rad": np.mod(base_angle + turn_adjust, 2.0 * np.pi),
        "wheel_fr_angle_rad": np.mod(base_angle - turn_adjust, 2.0 * np.pi),
        "wheel_rl_angle_rad": np.mod(base_angle + 0.5 * turn_adjust, 2.0 * np.pi),
        "wheel_rr_angle_rad": np.mod(base_angle - 0.5 * turn_adjust, 2.0 * np.pi),
        "front_steer_angle_rad": steer,
        "sensor_mast_pan_rad": sensor_pan,
        "suspension_travel_m": np.clip(suspension, -0.10, 0.10),
    }


def connectivity_and_gps(config: GeneratorConfig, robot_index: int, day_index: int, zone_idx: np.ndarray, elapsed_s: np.ndarray, rain: np.ndarray) -> Dict[str, np.ndarray]:
    """Return WiFi RSSI, GPS quality, and synthetic x-y position."""
    rng = rng_for(config, robot_index, day_index, "connectivity")
    multipath = ZONE_MULTIPATH_AMP[zone_idx] * np.sin(2.0 * np.pi * elapsed_s / 37.0 + robot_index)
    rssi = ZONE_RSSI_BASE[zone_idx] + multipath + 2.3 * np.sin(2.0 * np.pi * elapsed_s / 900.0 + day_index)
    rssi += rng.normal(0.0, 1.8, size=elapsed_s.size) - 2.0 * rain
    fade = rng.random(elapsed_s.size) < 0.003
    rssi = np.where(fade, rssi - rng.uniform(8.0, 18.0, size=elapsed_s.size), rssi)
    rssi = np.clip(rssi, -98.0, -35.0)
    gps_float = 5.0 - ZONE_GPS_PENALTY[zone_idx] - 0.8 * rain + rng.normal(0.0, 0.35, size=elapsed_s.size)
    gps_quality = np.clip(np.rint(gps_float), 0, 5).astype(float)
    gps_hdop = np.clip(0.55 + (5.0 - gps_quality) * 0.65 + 0.35 * rain + rng.normal(0.0, 0.08, size=elapsed_s.size), 0.45, 8.0)
    gps_sats = np.clip(np.rint(18 - 2.4 * (5 - gps_quality) + rng.normal(0.0, 1.2, size=elapsed_s.size)), 3, 22)
    x_m = 500.0 + 180.0 * np.sin(2.0 * np.pi * elapsed_s / 3600.0 + robot_index * 0.3)
    y_m = 500.0 + 135.0 * np.sin(2.0 * np.pi * elapsed_s / 2700.0 + robot_index * 0.5)
    return {
        "wifi_rssi_dbm": rssi,
        "wifi_fade_flag": fade.astype(int),
        "gps_fix_quality": gps_quality,
        "gps_hdop": gps_hdop,
        "gps_num_sats": gps_sats,
        "x_m": x_m,
        "y_m": y_m,
    }


def task_success(config: GeneratorConfig, robot_index: int, day_index: int, soc: np.ndarray, voltage: np.ndarray, gps_quality: np.ndarray, rssi: np.ndarray, currents: Mapping[str, np.ndarray], modes: np.ndarray, elapsed_s: np.ndarray) -> Dict[str, np.ndarray]:
    """Return task success heartbeat and checkpoint event flags."""
    rng = rng_for(config, robot_index, day_index, "task")
    checkpoint = ((elapsed_s + robot_index * 17.0) % (15 * 60)) < max(config.dt, 1.0)
    mean_current = np.mean(np.vstack([currents[k] for k in sorted(currents)]), axis=0)
    score = (
        0.30 * (soc > 15.0).astype(float)
        + 0.20 * (voltage > 42.0).astype(float)
        + 0.20 * (gps_quality >= 2.0).astype(float)
        + 0.15 * (rssi > -86.0).astype(float)
        + 0.15 * (mean_current < 36.0).astype(float)
    )
    mode_penalty = np.isin(modes, ["incident_response", "pursuit"]).astype(float) * 0.015
    prob = np.clip(0.985 + 0.014 * score - mode_penalty, 0.90, 0.9995)
    prob = np.where(modes == "charging", 1.0, prob)
    success = (rng.random(soc.size) < prob).astype(int)
    return {"checkpoint_event_flag": checkpoint.astype(int), "task_success_probability": prob, "task_success_flag": success}


def inject_noise(config: GeneratorConfig, robot_index: int, day_index: int, df: pd.DataFrame) -> pd.DataFrame:
    """Inject benign measurement noise and light missingness."""
    rng = rng_for(config, robot_index, day_index, "benign_noise")
    n = len(df)
    noise_flag = rng.random(n) < config.benign_noise_rate
    missing_flag = rng.random(n) < config.missing_rate
    if noise_flag.any():
        idx = np.where(noise_flag)[0]
        df.loc[df.index[idx], "wifi_rssi_dbm"] += rng.normal(0.0, 7.0, size=idx.size)
        df.loc[df.index[idx], "imu_accel_z_mps2"] += rng.normal(0.0, 0.55, size=idx.size)
        for col in ["motor_current_fl_a", "motor_current_fr_a", "motor_current_rl_a", "motor_current_rr_a"]:
            df.loc[df.index[idx], col] += rng.normal(0.0, 2.5, size=idx.size)
    if missing_flag.any():
        idx = np.where(missing_flag)[0]
        sensor_cols = ["wifi_rssi_dbm", "gps_fix_quality", "gps_hdop", "imu_accel_x_mps2", "imu_accel_y_mps2", "imu_gyro_z_rps"]
        chosen = rng.choice(sensor_cols, size=idx.size, replace=True)
        for row_idx, col in zip(idx, chosen):
            df.iat[row_idx, df.columns.get_loc(col)] = np.nan
    df["benign_noise_flag"] = noise_flag.astype(int)
    df["missingness_flag"] = missing_flag.astype(int)
    return df


def generate_partition(config: GeneratorConfig, robot_index: int, day_index: int, start_tick: int, n_ticks: int) -> pd.DataFrame:
    """Generate one deterministic robot-day partition."""
    ticks = np.arange(start_tick, start_tick + n_ticks, dtype=np.int64)
    elapsed_s = ticks / config.freq_hz
    seconds_of_day = np.mod(elapsed_s, SECONDS_PER_DAY)
    timestamps = build_timestamp_series(config, start_tick, n_ticks)

    soc, is_charging, _ = battery_profile(config, robot_index, day_index, elapsed_s)
    terrain_idx, terrain_name, friction, zone_name, zone_idx = terrain_and_zone(robot_index, day_index, elapsed_s)
    ambient_c, rain = weather(config, robot_index, day_index, seconds_of_day)
    modes = mode_schedule(config, robot_index, day_index, elapsed_s, is_charging, soc)
    speed = speed_profile(config, robot_index, day_index, modes, terrain_idx, rain, elapsed_s)
    currents, motor_temp_c = motor_current_and_temp(config, robot_index, day_index, speed, friction, rain, ambient_c, elapsed_s)
    voltage = battery_voltage(soc, is_charging, currents)
    imu = imu_signals(config, robot_index, day_index, speed, terrain_idx, rain, elapsed_s)
    joints = joint_angles(config, robot_index, day_index, speed, elapsed_s)
    conn = connectivity_and_gps(config, robot_index, day_index, zone_idx, elapsed_s, rain)
    task = task_success(config, robot_index, day_index, soc, voltage, conn["gps_fix_quality"], conn["wifi_rssi_dbm"], currents, modes, elapsed_s)

    df = pd.DataFrame(
        {
            "schema_version": SCHEMA_VERSION,
            "timestamp": timestamps.astype(str),
            "elapsed_s": elapsed_s,
            "day_index": day_index,
            "robot_id": robot_id(robot_index),
            "robot_index": robot_index,
            "mission_mode": modes,
            "terrain": terrain_name,
            "terrain_friction": friction,
            "location_zone": zone_name,
            "ambient_temp_c": ambient_c,
            "rain_intensity": rain,
            "speed_mps": speed,
            "battery_soc_pct": soc,
            "battery_voltage_v": voltage,
            "motor_temp_c": motor_temp_c,
            "is_charging": is_charging.astype(int),
        }
    )
    for group in (joints, imu, currents, conn, task):
        for col, values in group.items():
            df[col] = values
    df = inject_noise(config, robot_index, day_index, df)

    ordered_cols = [
        "schema_version", "timestamp", "elapsed_s", "day_index", "robot_id", "robot_index",
        "mission_mode", "terrain", "terrain_friction", "location_zone", "x_m", "y_m",
        "ambient_temp_c", "rain_intensity", "speed_mps",
        "wheel_fl_angle_rad", "wheel_fr_angle_rad", "wheel_rl_angle_rad", "wheel_rr_angle_rad",
        "front_steer_angle_rad", "sensor_mast_pan_rad", "suspension_travel_m",
        "imu_accel_x_mps2", "imu_accel_y_mps2", "imu_accel_z_mps2",
        "imu_gyro_x_rps", "imu_gyro_y_rps", "imu_gyro_z_rps",
        "motor_current_fl_a", "motor_current_fr_a", "motor_current_rl_a", "motor_current_rr_a",
        "motor_temp_c", "battery_voltage_v", "battery_soc_pct", "is_charging",
        "wifi_rssi_dbm", "wifi_fade_flag", "gps_fix_quality", "gps_hdop", "gps_num_sats",
        "checkpoint_event_flag", "task_success_probability", "task_success_flag",
        "benign_noise_flag", "missingness_flag",
    ]
    return df[ordered_cols]


def relative_partition_path(config: GeneratorConfig, robot_index: int, day_index: int) -> Path:
    suffix = {"csv": "csv", "csv.gz": "csv.gz", "parquet": "parquet"}[config.output_format]
    return Path(f"robot_id={robot_id(robot_index)}") / f"day={day_index:03d}" / f"telemetry.{suffix}"


def write_csv_deterministic(df: pd.DataFrame, path: Path, float_format: str, gzip_output: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gzip_output:
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as gz:
                with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                    df.to_csv(text, index=False, float_format=float_format, lineterminator="\n", na_rep="", quoting=csv.QUOTE_MINIMAL)
    else:
        df.to_csv(path, index=False, float_format=float_format, lineterminator="\n", na_rep="", quoting=csv.QUOTE_MINIMAL)


def write_partition(df: pd.DataFrame, config: GeneratorConfig, output_path: Path) -> None:
    if config.output_format == "csv":
        write_csv_deterministic(df, output_path, config.float_format, gzip_output=False)
    elif config.output_format == "csv.gz":
        write_csv_deterministic(df, output_path, config.float_format, gzip_output=True)
    elif config.output_format == "parquet":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {config.output_format}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def estimate_output(config: GeneratorConfig) -> Mapping[str, object]:
    rows = config.total_ticks * config.fleet_size
    partitions = sum(1 for _ in partition_ranges(config)) * config.fleet_size
    return {
        "fleet_size": config.fleet_size,
        "horizon_days": config.horizon_days,
        "freq_hz": config.freq_hz,
        "total_rows": rows,
        "partitions": partitions,
        "ticks_per_robot": config.total_ticks,
        "format": config.output_format,
        "output_dir": config.output_dir,
    }


def write_manifest(config: GeneratorConfig, records: List[Mapping[str, object]]) -> None:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "note": "Synthetic data. No real InGen operational telemetry.",
        "config": asdict(config),
        "estimate": estimate_output(config),
        "partitions": records,
    }
    Path(config.output_dir, "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def run_generation(config: GeneratorConfig) -> List[Mapping[str, object]]:
    validate_config(config)
    out = Path(config.output_dir)
    if out.exists():
        if config.force:
            shutil.rmtree(out)
        else:
            raise FileExistsError(f"Output directory already exists: {out}. Use --force to overwrite.")
    out.mkdir(parents=True, exist_ok=True)
    records: List[Mapping[str, object]] = []
    total = sum(1 for _ in partition_ranges(config)) * config.fleet_size
    counter = 0
    for r in range(config.fleet_size):
        for day_index, start_tick, n_ticks in partition_ranges(config):
            counter += 1
            rel = relative_partition_path(config, r, day_index)
            path = out / rel
            df = generate_partition(config, r, day_index, start_tick, n_ticks)
            write_partition(df, config, path)
            digest = sha256_file(path)
            records.append({
                "robot_id": robot_id(r),
                "robot_index": r,
                "day_index": day_index,
                "start_tick": start_tick,
                "n_ticks": n_ticks,
                "rows": len(df),
                "path": str(rel).replace("\\", "/"),
                "sha256": digest,
            })
            print(f"[{counter:>5}/{total}] wrote {rel} rows={len(df):,} sha256={digest[:12]}...")
    if config.write_manifest:
        write_manifest(config, records)
        print(f"Wrote manifest: {out / 'manifest.json'}")
    return records


def reproducibility_check(config: GeneratorConfig) -> bool:
    """Generate the same tiny partition twice and compare byte digests."""
    small = GeneratorConfig(
        seed=config.seed,
        fleet_size=1,
        horizon_days=min(config.horizon_days, 1.0 / 24.0),
        freq_hz=config.freq_hz,
        start_ts=config.start_ts,
        output_dir="unused",
        output_format=config.output_format,
        partition="robot_day",
        benign_noise_rate=config.benign_noise_rate,
        missing_rate=config.missing_rate,
        float_format=config.float_format,
        force=True,
        write_manifest=False,
    )
    if small.output_format == "parquet":
        print("WARNING: parquet metadata may vary by engine. Use csv or csv.gz for strict byte checks.")
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        c1 = GeneratorConfig(**{**asdict(small), "output_dir": d1})
        c2 = GeneratorConfig(**{**asdict(small), "output_dir": d2})
        n_ticks = min(c1.total_ticks, 600)
        df1 = generate_partition(c1, 0, 0, 0, n_ticks)
        df2 = generate_partition(c2, 0, 0, 0, n_ticks)
        p1 = Path(d1) / relative_partition_path(c1, 0, 0)
        p2 = Path(d2) / relative_partition_path(c2, 0, 0)
        write_partition(df1, c1, p1)
        write_partition(df2, c2, p2)
        h1, h2 = sha256_file(p1), sha256_file(p2)
    passed = h1 == h2
    print(f"Reproducibility check: {'PASS' if passed else 'FAIL'}")
    print(f"Digest 1: {h1}")
    print(f"Digest 2: {h2}")
    return passed


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic Aido Rover fleet telemetry.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=42, help="Global deterministic seed")
    parser.add_argument("--fleet-size", type=int, default=50, help="Number of simulated Rover units")
    parser.add_argument("--horizon-days", type=float, default=30.0, help="Simulation horizon in days")
    parser.add_argument("--freq-hz", type=float, default=1.0, help="Sampling frequency in Hz")
    parser.add_argument("--start-ts", type=str, default=DEFAULT_START_TS, help="UTC start timestamp")
    parser.add_argument("--output-dir", type=str, default="data/w03_rover_telemetry", help="Output directory")
    parser.add_argument("--format", dest="output_format", choices=["csv", "csv.gz", "parquet"], default="csv.gz", help="Output format")
    parser.add_argument("--partition", choices=["robot_day"], default="robot_day", help="Partition strategy")
    parser.add_argument("--benign-noise-rate", type=float, default=0.015, help="Baseline benign noise rate, recommended 0.01 to 0.02")
    parser.add_argument("--missing-rate", type=float, default=0.004, help="Light sensor missingness rate")
    parser.add_argument("--float-format", type=str, default="%.6f", help="Float format for deterministic CSV")
    parser.add_argument("--force", action="store_true", help="Overwrite output directory if it exists")
    parser.add_argument("--no-manifest", action="store_true", help="Do not write manifest.json")
    parser.add_argument("--dry-run", action="store_true", help="Print dataset size estimate and exit")
    parser.add_argument("--verify-reproducibility", action="store_true", help="Run byte-level reproducibility check before generation")
    parser.add_argument("--verify-only", action="store_true", help="Run reproducibility check and exit")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> GeneratorConfig:
    return GeneratorConfig(
        seed=args.seed,
        fleet_size=args.fleet_size,
        horizon_days=args.horizon_days,
        freq_hz=args.freq_hz,
        start_ts=args.start_ts,
        output_dir=args.output_dir,
        output_format=args.output_format,
        partition=args.partition,
        benign_noise_rate=args.benign_noise_rate,
        missing_rate=args.missing_rate,
        float_format=args.float_format,
        force=args.force,
        write_manifest=not args.no_manifest,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    validate_config(config)
    if args.dry_run:
        print(json.dumps(estimate_output(config), indent=2, sort_keys=True))
        return 0
    if args.verify_reproducibility or args.verify_only:
        if not reproducibility_check(config):
            return 2
        if args.verify_only:
            return 0
    print("Synthetic Aido Rover telemetry generation")
    print(json.dumps(estimate_output(config), indent=2, sort_keys=True))
    run_generation(config)
    print("Generation complete.")
    print("Reminder: this is synthetic telemetry, not real InGen operational data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
