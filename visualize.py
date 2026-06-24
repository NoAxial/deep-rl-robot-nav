"""
visualize.py -- Real-time Pygame visualisation of the trained RL agent.

Loads ``robot_nav_model.zip`` (and optional ``vec_normalize.pkl``), runs the
robot in the custom environment, and renders obstacles, goal, robot, and
LIDAR rays at 30 FPS.

Usage
-----
  py visualize.py                         # default config
  py visualize.py --config config.yaml    # custom config
  py visualize.py --model best_model/best_model.zip

Controls
--------
  ESC / close window   Quit
  R                    Reset episode
  SPACE                Pause / resume
  +/-                  Speed up / slow down (FPS)
  S                    Skip 1 episode
  Shift + S            Skip 10 episodes
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import pygame

from config import load_config
from batched_env import BatchedRobotEnv
from pytorch_sac import SAC

# =====================================================================
# Colour palette (dark premium theme)
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

PANEL_W = 260


# ── Helpers ──────────────────────────────────────────────────────────
def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def gradient_rect(surf, rect, c1, c2):
    x, y, w, h = (int(v) for v in rect)
    for row in range(h):
        color = lerp_color(c1, c2, row / max(h - 1, 1))
        pygame.draw.line(surf, color, (x, y + row), (x + w - 1, y + row))


# =====================================================================
# Main
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise trained RL agent")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, default="robot_nav_model")
    parser.add_argument("--skip-episodes", type=int, default=0, help="Fast-forward N episodes without rendering")
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_path = "models/sac_gpu_final.pt"
    if not Path(model_path).exists():
        print(f"[ERROR] No model found at {model_path}! Please run 'py train_agent_gpu.py' first.")
        sys.exit(1)

    # ---- Create batched env ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw_env = BatchedRobotEnv(config=cfg, num_envs=1, device=device)
    raw_env.set_difficulty(1.0)

    obs_dim = raw_env.obs_dim
    action_dim = raw_env.action_dim
    model = SAC(obs_dim, action_dim, device=device)
    model.load(model_path)
    print(f"[OK] Loaded PyTorch model: {model_path}")

    # ---- Pygame setup ----
    env_w, env_h = int(raw_env.world_w), int(raw_env.world_h)
    screen_w = env_w + PANEL_W
    pygame.init()
    screen = pygame.display.set_mode((screen_w, env_h))
    pygame.display.set_caption("Deep RL Robot Navigation -- SAC Agent")
    clock = pygame.time.Clock()

    font_title  = pygame.font.SysFont("Segoe UI", 22, bold=True)
    font_sub    = pygame.font.SysFont("Segoe UI", 14)
    font_stat   = pygame.font.SysFont("Segoe UI", 17)
    font_label  = pygame.font.SysFont("Segoe UI", 13)
    font_banner = pygame.font.SysFont("Segoe UI", 28, bold=True)

    # ---- State ----
    obs = raw_env.reset()
    episode       = 1
    total_reward  = 0.0
    step_count    = 0
    goals_reached = 0
    collisions    = 0
    timeouts      = 0
    ep_rewards: list[float] = []
    paused        = False
    fps           = 30
    flash_timer   = 0
    flash_color   = BG
    last_outcome  = ""
    last_action_str = "None"

    glow_surf = pygame.Surface((100, 100), pygame.SRCALPHA)
    robot_trail = deque(maxlen=60)
    # --- Setup tracking ---
    target_episode = args.skip_episodes
    running = True
    while running:
        # ── Events ──
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_r:
                    obs = raw_env.reset()
                    robot_trail.clear()
                    total_reward = 0.0
                    step_count = 0
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    fps = min(fps + 10, 120)
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    fps = max(fps - 10, 5)
                elif ev.key == pygame.K_s:
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_SHIFT:
                        target_episode = episode + 10
                    else:
                        target_episode = episode + 1

        if paused:
            clock.tick(fps)
            continue


        # ── Agent step ──
        action = model.select_action(obs, evaluate=True)
        obs, reward, dones, infos = raw_env.step(action)
        
        last_action_str = f"V:{float(action[0, 0]):+.2f} W:{float(action[0, 1]):+.2f}"
        
        total_reward += float(reward[0].item())
        step_count += 1

        if dones[0]:
            ep_rewards.append(total_reward)
            robot_trail.clear()
            
            # Reconstruct collision/goal logic since info dict is empty in batched env
            if float(reward[0].item()) <= -40.0:  # heuristic for collision reward (now -50.0)
                collisions += 1
                flash_color = FAIL_RED
                last_outcome = "COLLISION!"
            elif float(reward[0].item()) >= 45.0: # heuristic for goal reward (now 50.0)
                goals_reached += 1
                flash_color = SUCCESS_GRN
                last_outcome = "GOAL REACHED!"
            else:
                timeouts += 1
                flash_color = ACCENT
                last_outcome = "TIMEOUT"
                
            flash_timer = 40
            episode += 1
            total_reward = 0.0
            step_count = 0
            # BatchedEnv auto-resets, obs already updated

        if episode < target_episode:
            continue

        # ==============================================================
        # DRAW
        # ==============================================================
        screen.fill(BG)

        # Grid (Soft crosshairs)
        for gx in range(0, env_w, 40):
            for gy in range(0, env_h, 40):
                pygame.draw.line(screen, GRID, (gx - 2, gy), (gx + 2, gy))
                pygame.draw.line(screen, GRID, (gx, gy - 2), (gx, gy + 2))
        pygame.draw.rect(screen, WALL, (0, 0, env_w, env_h), 3)

        # Obstacles
        for i in range(raw_env.num_obstacles[0]):
            ox = float(raw_env.obstacles[0, i, 0].item())
            oy = float(raw_env.obstacles[0, i, 1].item())
            ow = float(raw_env.obstacles[0, i, 2].item())
            oh = float(raw_env.obstacles[0, i, 3].item())
            
            pygame.draw.rect(screen, (6, 6, 14),
                             (ox + 4, oy + 4, ow, oh), border_radius=5)
            gradient_rect(screen, (ox, oy, ow, oh), OBSTACLE, OBS_DARK)
            pygame.draw.rect(screen, OBS_BORDER,
                             (ox, oy, ow, oh), 2, border_radius=5)

        # Goal glow
        g = (int(raw_env.goal_pos[0, 0].item()), int(raw_env.goal_pos[0, 1].item()))
        pulse = 0.5 + 0.5 * math.sin(pygame.time.get_ticks() / 250.0)
        glow_r = int(raw_env.goal_radius + 18 * pulse)
        glow_surf.fill((0, 0, 0, 0))
        for r in range(glow_r, int(raw_env.goal_radius), -2):
            alpha = int(35 * (1 - (r - raw_env.goal_radius) /
                               max(glow_r - raw_env.goal_radius, 1)))
            pygame.draw.circle(glow_surf, (*GOAL_GREEN, alpha), (50, 50), r)
        screen.blit(glow_surf, (g[0] - 50, g[1] - 50),
                    special_flags=pygame.BLEND_RGBA_ADD)
        pygame.draw.circle(screen, GOAL_GREEN, g, int(raw_env.goal_radius))
        pygame.draw.circle(screen, (255, 255, 255), g,
                           int(raw_env.goal_radius), 2)
        lbl = font_label.render("GOAL", True, (255, 255, 255))
        screen.blit(lbl, (g[0] - lbl.get_width() // 2,
                          g[1] - lbl.get_height() // 2))

        # LIDAR rays
        rx = int(raw_env.robot_pos[0, 0].item())
        ry = int(raw_env.robot_pos[0, 1].item())
        ns = raw_env.num_sensors
        
        # Trail
        robot_trail.append((rx, ry))
        if len(robot_trail) > 1:
            pts = list(robot_trail)
            tsurf = pygame.Surface((env_w, env_h), pygame.SRCALPHA)
            for i in range(len(pts) - 1):
                alpha = int(255 * (i / len(pts)))
                col = (*ROBOT_BLUE, alpha)
                pygame.draw.line(tsurf, col, pts[i], pts[i+1], 3)
            screen.blit(tsurf, (0, 0))

        if ns > 0:
            for i in range(ns):
                angle = raw_env.robot_angle[0].item() + raw_env.ray_angles[i].item()
                # Extract the newest LIDAR frame from the frame stacking buffer
                dist = raw_env.obs_buffer[0, -1, i].item() * raw_env.sensor_range
                ex = rx + int(dist * np.cos(angle))
                ey = ry + int(dist * np.sin(angle))
                t = max(0.0, min(1.0, dist / raw_env.sensor_range))
                color = lerp_color(FAIL_RED, (241, 196, 15), t)
                pygame.draw.line(screen, color, (rx, ry), (ex, ey), 1)
                pygame.draw.circle(screen, color, (ex, ey), 2)

        # Robot (Sleek drone shape)
        ra = raw_env.robot_angle[0].item()
        rr = raw_env.robot_radius + 4
        # Points for a triangle/wedge
        p1 = (rx + int(rr * np.cos(ra)), ry + int(rr * np.sin(ra)))
        p2 = (rx + int(rr * 0.8 * np.cos(ra + 2.5)), ry + int(rr * 0.8 * np.sin(ra + 2.5)))
        p3 = (rx + int(rr * 0.5 * np.cos(ra + np.pi)), ry + int(rr * 0.5 * np.sin(ra + np.pi)))
        p4 = (rx + int(rr * 0.8 * np.cos(ra - 2.5)), ry + int(rr * 0.8 * np.sin(ra - 2.5)))
        pygame.draw.polygon(screen, ROBOT_RING, [p1, p2, p3, p4], 2)
        pygame.draw.polygon(screen, ROBOT_BLUE, [p1, p2, p3, p4])
        
        # Heading dot
        pygame.draw.circle(screen, HEADING_WHT, p1, 3)

        # Flash overlay
        if flash_timer > 0:
            flash_timer -= 1
            alpha = int(90 * (flash_timer / 40))
            overlay = pygame.Surface((env_w, env_h), pygame.SRCALPHA)
            overlay.fill((*flash_color, alpha))
            screen.blit(overlay, (0, 0))
            txt = font_banner.render(last_outcome, True, (255, 255, 255))
            screen.blit(txt, (env_w // 2 - txt.get_width() // 2,
                              env_h // 2 - txt.get_height() // 2))

        # ==============================================================
        # SIDE PANEL
        # ==============================================================
        px = env_w
        pygame.draw.rect(screen, PANEL_BG, (px, 0, PANEL_W, env_h))
        pygame.draw.line(screen, ACCENT, (px, 0), (px, env_h), 2)

        t1 = font_title.render("RL Navigation", True, TEXT)
        screen.blit(t1, (px + 18, 14))
        t2 = font_sub.render("Batched GPU SAC  |  Real-time", True, ACCENT)
        screen.blit(t2, (px + 18, 42))
        pygame.draw.line(screen, GRID, (px + 18, 65), (px + PANEL_W - 18, 65))

        stats = [
            ("Episode",       str(episode)),
            ("Step",          f"{step_count} / {raw_env.max_steps}"),
            ("Reward",        f"{total_reward:+.1f}"),
            ("Action",        last_action_str),
            ("FPS",           str(fps)),
            ("",              ""),
            ("Goals",         str(goals_reached)),
            ("Collisions",    str(collisions)),
            ("Timeouts",      str(timeouts)),
        ]
        y = 78
        for label, value in stats:
            if not label:
                y += 6
                continue
            sl = font_stat.render(label, True, TEXT_DIM)
            sv = font_stat.render(value, True, TEXT)
            screen.blit(sl, (px + 18, y))
            screen.blit(sv, (px + PANEL_W - 18 - sv.get_width(), y))
            y += 26

        # LIDAR bars
        y += 8
        pygame.draw.line(screen, GRID, (px + 18, y), (px + PANEL_W - 18, y))
        y += 10
        screen.blit(font_stat.render("LIDAR Sensors", True, TEXT), (px + 18, y))
        y += 28
        bar_w = (PANEL_W - 50) // (ns or 1)
        for i in range(ns):
            val = raw_env.obs_buffer[0, -1, i].item()
            bar_h = int(val * 55)
            bx = px + 20 + i * (bar_w + 2)
            by = y + 55 - bar_h
            col = lerp_color(FAIL_RED, (241, 196, 15), val)
            pygame.draw.rect(screen, col, (bx, by, bar_w - 2, bar_h),
                             border_radius=2)
            deg_per = 360 // ns
            deg = font_label.render(f"{i * deg_per}", True, TEXT_DIM)
            screen.blit(deg, (bx, y + 60))

        # Performance
        y += 90
        pygame.draw.line(screen, GRID, (px + 18, y), (px + PANEL_W - 18, y))
        y += 10
        screen.blit(font_stat.render("Performance", True, TEXT), (px + 18, y))
        y += 26
        total_ep = max(episode - 1, 1)
        success_pct = goals_reached / total_ep * 100
        avg_rew = np.mean(ep_rewards[-20:]) if ep_rewards else 0.0
        for label, value in [("Avg Reward (20)", f"{avg_rew:+.1f}"),
                             ("Success Rate", f"{success_pct:.0f} %")]:
            sl = font_sub.render(label, True, TEXT_DIM)
            sc = SUCCESS_GRN if "Rate" in label and success_pct >= 50 else TEXT
            sv = font_sub.render(value, True, sc)
            screen.blit(sl, (px + 18, y))
            screen.blit(sv, (px + PANEL_W - 18 - sv.get_width(), y))
            y += 22

        # Controls
        hint = font_label.render("ESC quit | R reset | SPACE pause | +/- FPS", True, TEXT_DIM)
        hint2 = font_label.render("S skip | Shift+S skip 10", True, TEXT_DIM)
        screen.blit(hint, (px + 18, env_h - 40))
        screen.blit(hint2, (px + 18, env_h - 20))

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
    print("Visualisation closed.")


if __name__ == "__main__":
    main()
