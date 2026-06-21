"""
train_visual.py -- Watch the PPO agent learn in real-time.

This script runs the training loop but uses a custom Pygame callback to render
the environment and live metrics (reward curve, success rate) side-by-side.

Usage
-----
  py train_visual.py
  py train_visual.py --config configs/hard.yaml

Controls
--------
  F     Toggle Fast mode (disables rendering for max speed)
  S     Toggle Slow mode (renders every step at 30 FPS)
  SPACE Pause/Resume
  ESC   Save & Quit
"""

from __future__ import annotations

import argparse
import collections
import math
import os
import sys
from pathlib import Path

import numpy as np
import pygame
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor

from config import load_config
from robot_env import RobotNavEnv


# =====================================================================
# Colour palette
# =====================================================================
BG          = (12, 12, 24)
GRID        = (22, 22, 40)
WALL        = (50, 50, 80)
OBSTACLE    = (220, 65, 55)
OBS_DARK    = (160, 40, 30)
OBS_BORDER  = (180, 50, 40)
GOAL_GREEN  = (46, 204, 113)
ROBOT_BLUE  = (52, 152, 219)
ROBOT_RING  = (41, 128, 185)
HEADING_WHT = (240, 240, 255)
TEXT        = (220, 225, 230)
TEXT_DIM    = (120, 125, 140)
PANEL_BG    = (18, 18, 34)
ACCENT      = (155, 89, 182)
SUCCESS_GRN = (39, 174, 96)
FAIL_RED    = (235, 77, 65)

PANEL_W = 350


def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def gradient_rect(surf, rect, c1, c2):
    x, y, w, h = (int(v) for v in rect)
    for row in range(h):
        color = lerp_color(c1, c2, row / max(h - 1, 1))
        pygame.draw.line(surf, color, (x, y + row), (x + w - 1, y + row))


# =====================================================================
# Pygame Rendering Callback
# =====================================================================
class LiveVizCallback(BaseCallback):
    def __init__(self, raw_envs: list, verbose: int = 1):
        super().__init__(verbose)
        self.raw_envs = raw_envs
        
        cfg = self.raw_envs[0].config
        self.world_w = cfg["world"]["width"]
        self.world_h = cfg["world"]["height"]
        
        self.panel_w = 400
        self.screen_w = self.world_w + self.panel_w
        self.screen_h = self.world_h
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("Deep RL Navigation -- Live Training Viz")
        self.clock = pygame.time.Clock()

        self.font_title  = pygame.font.SysFont("Segoe UI", 22, bold=True)
        self.font_sub    = pygame.font.SysFont("Segoe UI", 14)
        self.font_stat   = pygame.font.SysFont("Segoe UI", 17)
        self.font_label  = pygame.font.SysFont("Segoe UI", 12)
        self.font_small  = pygame.font.SysFont("Segoe UI", 10)

        self.fast_mode = False
        self.paused = False
        self.fps = 60

        self.ep_rewards = collections.deque(maxlen=200)
        self.ep_lengths = collections.deque(maxlen=200)
        self.success_history = collections.deque(maxlen=200)
        self.episodes_done = 0
        self.goals_reached = 0

    def _on_step(self) -> bool:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return False
                elif ev.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif ev.key == pygame.K_f:
                    self.fast_mode = not self.fast_mode
                    self.fps = 0 if self.fast_mode else 60
                elif ev.key == pygame.K_s:
                    self.fast_mode = False
                    self.fps = 30

        while self.paused:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    return False
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                    self.paused = False
            self.clock.tick(15)

        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for idx, done in enumerate(dones):
            if done and len(infos) > idx and "episode" in infos[idx]:
                self.episodes_done += 1
                r = infos[idx]["episode"]["r"]
                l = infos[idx]["episode"]["l"]
                self.ep_rewards.append(r)
                self.ep_lengths.append(l)
                is_success = r > 50
                if is_success:
                    self.goals_reached += 1
                self.success_history.append(1.0 if is_success else 0.0)

        if self.fast_mode and self.num_timesteps % 100 != 0:
            return True

        self._draw_screen()
        if self.fps > 0:
            self.clock.tick(self.fps)
        return True

    def _draw_screen(self):
        self.screen.fill(BG)

        n = len(self.raw_envs)
        self.grid_cols = math.ceil(math.sqrt(n))
        self.grid_rows = math.ceil(n / self.grid_cols)
        self.sub_w = self.world_w // self.grid_cols
        self.sub_h = self.world_h // self.grid_rows

        for i, env in enumerate(self.raw_envs):
            env.render_mode = "rgb_array"
            frame = env.render()
            if frame is not None:
                surf = pygame.surfarray.make_surface(np.swapaxes(frame, 0, 1))
                if n > 1:
                    surf = pygame.transform.smoothscale(surf, (self.sub_w, self.sub_h))
                col = i % self.grid_cols
                row = i // self.grid_cols
                x = col * self.sub_w
                y = row * self.sub_h
                self.screen.blit(surf, (x, y))
                if n > 1:
                    pygame.draw.rect(self.screen, GRID, (x, y, self.sub_w, self.sub_h), 2)
                    text_surf = self.font_small.render(f"Env {i}", True, TEXT_DIM)
                    self.screen.blit(text_surf, (x + 10, y + 10))

        px = self.world_w
        pygame.draw.rect(self.screen, PANEL_BG, (px, 0, self.panel_w, self.screen_h))
        pygame.draw.line(self.screen, ACCENT, (px, 0), (px, self.screen_h), 2)

        t1 = self.font_title.render("Live Training Dashboard", True, TEXT)
        self.screen.blit(t1, (px + 18, 14))
        
        mode_text = "FAST MODE" if self.fast_mode else "NORMAL SPEED"
        mode_col = FAIL_RED if self.fast_mode else SUCCESS_GRN
        t2 = self.font_sub.render(f"Mode: {mode_text}", True, mode_col)
        self.screen.blit(t2, (px + 18, 42))

        avg_rew = np.mean(self.ep_rewards) if self.ep_rewards else 0.0
        avg_len = np.mean(self.ep_lengths) if self.ep_lengths else 0.0
        succ_rate = np.mean(self.success_history) * 100 if self.success_history else 0.0

        stats = [
            ("Timesteps", f"{self.num_timesteps:,}"),
            ("Episodes", f"{self.episodes_done:,}"),
            ("Avg Reward", f"{avg_rew:+.1f}"),
            ("Avg Length", f"{avg_len:.0f} steps"),
            ("Success Rate", f"{succ_rate:.1f} %"),
        ]
        
        y = 78
        for label, value in stats:
            sl = self.font_stat.render(label, True, TEXT_DIM)
            sc = SUCCESS_GRN if ("Success" in label and succ_rate > 50) else TEXT
            sv = self.font_stat.render(value, True, sc)
            self.screen.blit(sl, (px + 18, y))
            self.screen.blit(sv, (px + self.panel_w - 18 - sv.get_width(), y))
            y += 26

        self._draw_chart("Reward Curve (last 200)", self.ep_rewards, px + 18, y + 20, self.panel_w - 36, 100, color=ROBOT_BLUE)
        self._draw_chart("Episode Length", self.ep_lengths, px + 18, y + 160, self.panel_w - 36, 100, color=ACCENT)

        hint = self.font_label.render("F: Fast | S: Slow | SPACE: Pause | ESC: Save & Quit", True, TEXT_DIM)
        self.screen.blit(hint, (px + 18, self.screen_h - 30))

        pygame.display.flip()

    def _draw_chart(self, title, data_q, x, y, w, h, color):
        tl = self.font_sub.render(title, True, TEXT)
        self.screen.blit(tl, (x, y - 20))
        pygame.draw.rect(self.screen, (25, 25, 45), (x, y, w, h), border_radius=4)
        pygame.draw.rect(self.screen, GRID, (x, y, w, h), 1, border_radius=4)
        if not data_q: return
        data = list(data_q)
        max_val = max(data) if max(data) > 0 else 1
        min_val = min(data) if min(data) < 0 else 0
        range_val = max_val - min_val if (max_val - min_val) != 0 else 1
        pts = [(x + (i / max(len(data) - 1, 1)) * w, y + h - ((val - min_val) / range_val) * h) for i, val in enumerate(data)]
        if len(pts) > 1: pygame.draw.lines(self.screen, color, False, pts, 2)
        elif len(pts) == 1: pygame.draw.circle(self.screen, color, pts[0], 2)

    def _on_training_end(self) -> None:
        pygame.quit()


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Live visual training of RL agent")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    t_cfg = cfg["training"]

    def _make_env():
        env = RobotNavEnv(config=cfg)
        return Monitor(env)

    n_envs = int(t_cfg.get("n_envs", 4))
    vec_env = DummyVecEnv([_make_env for _ in range(n_envs)])
    
    # Wrap with Frame Stacking for temporal awareness (future positioning)
    vec_env = VecFrameStack(vec_env, n_stack=4)

    # Load normalisation stats if available
    norm_path = Path("vec_normalize.pkl")
    if norm_path.exists():
        vec_env = VecNormalize.load(str(norm_path), vec_env)
        vec_env.training = True
        print("[OK] VecNormalize stats loaded from disk to continue training")
    elif t_cfg["normalize_obs"] or t_cfg["normalize_reward"]:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=t_cfg["normalize_obs"],
            norm_reward=t_cfg["normalize_reward"],
            clip_obs=10.0,
        )
        print(f"[OK] VecNormalize initialized (obs={t_cfg['normalize_obs']}, reward={t_cfg['normalize_reward']})")

    # Access the inner DummyVecEnv
    inner_vec = vec_env
    while hasattr(inner_vec, 'venv'):
        inner_vec = inner_vec.venv
    
    raw_envs = []
    for i in range(n_envs):
        env_i = inner_vec.envs[i]
        raw_envs.append(env_i.unwrapped if hasattr(env_i, 'unwrapped') else env_i)
    
    from train_agent import _get_activation
    from custom_policy import CustomCombinedExtractor
    
    n_sensors = int(cfg["robot"]["num_sensors"])
    
    policy_kwargs = {
        "activation_fn": _get_activation(t_cfg["activation"]),
        "net_arch": dict(pi=list(t_cfg["net_arch_pi"]), qf=list(t_cfg["net_arch_vf"])),
        "features_extractor_class": CustomCombinedExtractor,
        "features_extractor_kwargs": dict(features_dim=256, num_sensors=n_sensors, n_stack=4)
    }
    
    model_path = "robot_nav_model.zip"
    if os.path.exists(model_path):
        print(f"[INFO] Loading existing model from {model_path} to continue training...")
        model = SAC.load(model_path, env=vec_env, device="cpu")
    else:
        model = SAC(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            device="cpu",
            learning_rate=float(t_cfg["learning_rate"]),
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

    viz_callback = LiveVizCallback(raw_envs=raw_envs)
    from curriculum import PerformanceCurriculumCallback
    curr_callback = PerformanceCurriculumCallback(raw_envs=raw_envs, window_size=50)

    try:
        model.learn(total_timesteps=int(t_cfg["total_timesteps"]), callback=[curr_callback, viz_callback])
    except KeyboardInterrupt:
        print("[INFO] Training interrupted.")
        
    model.save("robot_nav_model_live")
    if isinstance(vec_env, VecNormalize):
        vec_env.save("vec_normalize_live.pkl")

    print("[DONE] Live training finished. Model saved as 'robot_nav_model_live.zip'")

if __name__ == "__main__":
    main()
