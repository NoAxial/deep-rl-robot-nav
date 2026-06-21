"""
train_agent.py -- Train a PPO agent on the custom RobotNavEnv.

Improvements over the baseline:
  - YAML-driven configuration (--config flag)
  - VecNormalize for observation & reward normalisation
  - Custom network architecture (128x128x64, ReLU)
  - Learning-rate & clip-range linear scheduling
  - Early stopping via EvalCallback
  - Best-model auto-save
  - 200k timestep budget (stops early if solved)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from functools import partial

import torch
from stable_baselines3 import SAC
from custom_policy import CustomCombinedExtractor
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack

from config import load_config
from robot_env import RobotNavEnv


# ==================================================================
# Custom callback -- periodic summary
# ==================================================================
class MetricsCallback(BaseCallback):
    """Print a one-line summary every *freq* timesteps."""

    def __init__(self, freq: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.freq = freq
        self._last = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.freq:
            infos = self.locals.get("infos", [])
            ep_rews = [i["episode"]["r"] for i in infos if "episode" in i]
            if ep_rews:
                avg = sum(ep_rews) / len(ep_rews)
                print(f"  [T] {self.num_timesteps:>8,} steps | avg reward: {avg:+.1f}")
            self._last = self.num_timesteps
        return True


# ==================================================================
# Helpers
# ==================================================================
def _make_env(config: dict):
    """Factory that returns a callable producing a RobotNavEnv."""
    def _init():
        return RobotNavEnv(config=config)
    return _init


def _get_activation(name: str):
    return {"relu": torch.nn.ReLU, "tanh": torch.nn.Tanh}.get(name, torch.nn.ReLU)


def _get_schedule(kind: str, initial: float):
    if kind == "linear":
        def func(progress_remaining: float) -> float:
            return progress_remaining * initial
        return func
    return initial  # constant


# ==================================================================
# Main
# ==================================================================
def main() -> None:
    # ---- CLI ----
    parser = argparse.ArgumentParser(description="Train RL robot navigation agent")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (default: built-in defaults)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    t_cfg = cfg["training"]

    print("")
    print("=" * 60)
    print("  Deep RL Autonomous Mobile Robot Navigation -- Training")
    print("=" * 60)

    # ---- 1. Validate ----
    print("\n[1/5] Validating environment ...")
    test_env = RobotNavEnv(config=cfg)
    check_env(test_env, warn=True)
    print("      [OK] Environment passed all checks.")
    print(f"      Observation space: {test_env.observation_space.shape}")
    print(f"      Action space:      {test_env.action_space}")

    # ---- 2. Vectorised + normalised env ----
    n_envs = int(t_cfg["n_envs"])
    print(f"\n[2/5] Creating {n_envs} parallel environments ...")
    vec_env = make_vec_env(_make_env(cfg), n_envs=n_envs)

    # Wrap with Frame Stacking for temporal awareness (future positioning)
    vec_env = VecFrameStack(vec_env, n_stack=4)
    print("      [OK] VecFrameStack enabled (n_stack=4)")

    # 2. Add VecNormalize to normalise observations and rewards
    if t_cfg["normalize_obs"] or t_cfg["normalize_reward"]:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=t_cfg["normalize_obs"],
            norm_reward=t_cfg["normalize_reward"],
            clip_obs=10.0,
        )
        print(f"      [OK] VecNormalize enabled (obs={t_cfg['normalize_obs']}, reward={t_cfg['normalize_reward']})")
    else:
        print("      [OK] Normalisation disabled")

    # ---- 3. Build SAC ----
    print("\n[3/5] Building SAC agent ...")
    
    # Custom Feature Extractor args
    n_sensors = int(cfg["robot"]["num_sensors"])
    policy_kwargs = dict(
        net_arch=dict(
            pi=list(t_cfg["net_arch_pi"]),
            qf=list(t_cfg["net_arch_vf"]),
        ),
        activation_fn=_get_activation(t_cfg["activation"]),
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(features_dim=256, n_stack=4)
    )

    lr = float(t_cfg["learning_rate"])

    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        device=t_cfg.get("device", "auto"),
        learning_rate=lr,
        buffer_size=int(t_cfg["buffer_size"]),
        batch_size=int(t_cfg["batch_size"]),
        tau=float(t_cfg["tau"]),
        gamma=float(t_cfg["gamma"]),
        train_freq=int(t_cfg["train_freq"]),
        gradient_steps=int(t_cfg["gradient_steps"]),
        learning_starts=int(t_cfg["learning_starts"]),
        ent_coef=t_cfg["ent_coef"],
        seed=42,
        policy_kwargs=policy_kwargs,
    )
    print(f"      [OK] Network: pi={t_cfg['net_arch_pi']}  qf={t_cfg['net_arch_vf']}")
    print(f"      [OK] Activation: {t_cfg['activation'].upper()}")
    print(f"      [OK] LR: {lr}")

    # ---- 4. Callbacks ----
    print("\n[4/5] Setting up callbacks ...")
    callbacks = [MetricsCallback(freq=10_000)]

    # Separate eval env (also normalised with training stats)
    eval_env = make_vec_env(_make_env(cfg), n_envs=1)
    
    # Wrap with Frame Stacking
    eval_env = VecFrameStack(eval_env, n_stack=4)

    if t_cfg["normalize_obs"]:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            training=False,
        )

    # Custom callback: sync eval norm stats + save checkpoints periodically
    class SyncAndSaveCallback(BaseCallback):
        """Syncs eval VecNormalize stats from training env before each eval,
           saves vec_normalize.pkl whenever a new best model is saved,
           and periodically saves a 'latest' checkpoint every `save_freq` steps."""
        def __init__(self, model_ref, train_env, eval_env, save_path,
                     save_freq=10_000, verbose=0):
            super().__init__(verbose)
            self.model_ref = model_ref
            self.train_env = train_env
            self.eval_env = eval_env
            self.save_path = Path(save_path)
            self.save_freq = save_freq
            self._last_best = None
            self._last_periodic = 0

        def _on_step(self) -> bool:
            # Sync running mean/var from training env to eval env
            if isinstance(self.train_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
                self.eval_env.obs_rms = self.train_env.obs_rms
                self.eval_env.ret_rms = self.train_env.ret_rms

            # Check if a new best_model was just saved by EvalCallback
            best_model_path = self.save_path / "best_model.zip"
            if best_model_path.exists():
                mtime = best_model_path.stat().st_mtime
                if self._last_best is None or mtime > self._last_best:
                    self._last_best = mtime
                    # Save vec_normalize.pkl alongside best_model
                    if isinstance(self.train_env, VecNormalize):
                        self.train_env.save(str(self.save_path / "vec_normalize.pkl"))
                        if self.verbose:
                            print(f"      [SAVE] vec_normalize.pkl synced with best_model")

            # Periodic save of latest model + norm stats every save_freq steps
            if self.num_timesteps - self._last_periodic >= self.save_freq:
                self._last_periodic = self.num_timesteps
                self.model.save("robot_nav_model")
                if isinstance(self.train_env, VecNormalize):
                    self.train_env.save("vec_normalize.pkl")
                if self.verbose:
                    print(f"      [CHECKPOINT] Saved latest model @ {self.num_timesteps:,} steps")

            return True

    sync_cb = SyncAndSaveCallback(
        model_ref=None,  # will be set by SB3 via self.model
        train_env=vec_env, eval_env=eval_env,
        save_path="./best_model/", save_freq=10_000, verbose=1
    )
    callbacks.append(sync_cb)

    stop_cb = None
    if t_cfg["early_stopping"]:
        stop_cb = StopTrainingOnRewardThreshold(
            reward_threshold=float(t_cfg["reward_threshold"]), verbose=1
        )
        print(f"      [OK] Early stopping at reward >= {t_cfg['reward_threshold']}")

    eval_cb = EvalCallback(
        eval_env,
        callback_on_new_best=stop_cb,
        eval_freq=max(int(t_cfg["eval_freq"]) // n_envs, 1),
        n_eval_episodes=int(t_cfg["n_eval_episodes"]),
        best_model_save_path="./best_model/",
        verbose=1,
    )
    callbacks.append(eval_cb)
    print(f"      [OK] Eval every {t_cfg['eval_freq']} steps ({t_cfg['n_eval_episodes']} episodes)")

    # Curriculum callback
    from curriculum import PerformanceCurriculumCallback
    # Access through wrappers -> DummyVecEnv -> Monitor -> RobotNavEnv
    inner_vec = vec_env
    while hasattr(inner_vec, 'venv'):
        inner_vec = inner_vec.venv
        
    raw_envs = []
    for env_i in inner_vec.envs:
        if hasattr(env_i, 'unwrapped'):
            raw_envs.append(env_i.unwrapped)
        else:
            raw_envs.append(env_i)
            
    curr_callback = PerformanceCurriculumCallback(raw_envs=raw_envs, window_size=50, verbose=1)
    callbacks.append(curr_callback)

    # ---- 5. Train ----
    print(f"\n[5/5] Training for up to {t_cfg['total_timesteps']:,} timesteps ...")
    print("-" * 60)

    t0 = time.perf_counter()
    model.learn(total_timesteps=int(t_cfg["total_timesteps"]), callback=callbacks)
    elapsed = time.perf_counter() - t0

    # ---- Save ----
    model.save("robot_nav_model")
    if isinstance(vec_env, VecNormalize):
        vec_env.save("vec_normalize.pkl")
        print("      Saved normalisation stats -> vec_normalize.pkl")

    print("-" * 60)
    print(f"\n[DONE] Training complete in {elapsed:.1f} s")
    print("       Model saved -> robot_nav_model.zip")
    print("\n       Next step:  py visualize.py")


if __name__ == "__main__":
    main()
