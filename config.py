"""
config.py -- Configuration loader for the robot navigation project.

Provides a single ``load_config(path)`` entry-point that returns a fully
populated config dict.  Missing keys in user YAML files are filled from
built-in defaults so every module always sees a complete config.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# ======================================================================
# Default configuration (used when no YAML is provided or for missing keys)
# ======================================================================
DEFAULT_CONFIG: dict[str, Any] = {
    "world": {
        "width": 800,
        "height": 600,
        "num_obstacles_min": 3,
        "num_obstacles_max": 5,
        "obstacle_width_min": 40,
        "obstacle_width_max": 120,
        "obstacle_height_min": 30,
        "obstacle_height_max": 90,
        "dynamic_obstacles": False,
        "dynamic_speed_min": 1.0,
        "dynamic_speed_max": 3.0,
    },
    "robot": {
        "radius": 12,
        "step_size": 8.0,
        "turn_angle": 20,          # degrees
        "num_sensors": 32,         # HD LIDAR
        "sensor_range": 200,
        "action_mode": "car_continuous",
    },
    "goal": {
        "radius": 20,
    },
    "rewards": {
        "goal": 200.0,
        "collision": -100.0,
        "step_penalty": -0.05,
        "velocity_reward": 0.5,
        "clear_path_reward": 0.05,
        "distance_shaping": 0.0,      
        "heading_bonus": 0.0,         
        "proximity_penalty": 0.2,
        "proximity_threshold": 0.1,
        "speed_bonus": 50.0,
        "spin_penalty": 0.05,
        "jerk_penalty": 0.02,
    },
    "training": {
        "algorithm": "SAC",
        "total_timesteps": 400_000,
        "n_envs": 4,
        "learning_rate": 0.001,
        "buffer_size": 100_000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 8,
        "gradient_steps": 4,
        "learning_starts": 1000,
        "ent_coef": 0.05,
        "device": "auto",
        "net_arch_pi": [64, 64],
        "net_arch_vf": [64, 64],
        "lr_schedule": "constant",
        "normalize_obs": True,
        "normalize_reward": True,
        "early_stopping": True,
        "reward_threshold": 120.0,
        "eval_freq": 5_000,
        "n_eval_episodes": 20,
        "activation": "relu",           # "relu" or "tanh"
    },
    "episode": {
        "max_steps": 500,
    },
}


# ======================================================================
# Helpers
# ======================================================================
def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (non-destructive)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and return a fully populated config dict.

    Parameters
    ----------
    path : str or Path, optional
        Path to a user YAML file.  Any keys present override defaults;
        missing keys are filled from ``DEFAULT_CONFIG``.
        If *None* or the file does not exist, pure defaults are returned.
    """
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as fh:
                user = yaml.safe_load(fh) or {}
            config = _deep_merge(config, user)
            print(f"[config] Loaded: {p}")
        else:
            print(f"[config] File not found: {p} -- using defaults")
    else:
        print("[config] Using default configuration")
    return config
