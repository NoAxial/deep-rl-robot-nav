"""
analytics.py -- Evaluate a trained agent and generate visual analytics.

Runs 100 evaluation episodes and generates a dashboard of plots:
  1. Trajectory Heatmap
  2. Reward Source Breakdown
  3. Episode Length Distribution
  4. Collision Scatter Plot

Usage:
  py analytics.py --model robot_nav_model.zip
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack

from config import load_config
from robot_env import RobotNavEnv


def generate_analytics(model_path: str, config_path: str | None = None, num_episodes: int = 100):
    cfg = load_config(config_path)

    # Disable dynamic obstacles during analytics to get a clear heatmap of static navigation
    cfg["world"]["dynamic_obstacles"] = False

    raw_env = RobotNavEnv(config=cfg)
    vec_env = DummyVecEnv([lambda: raw_env])

    # Wrap with Frame Stacking
    vec_env = VecFrameStack(vec_env, n_stack=4)

    norm_path = Path("vec_normalize.pkl")
    if norm_path.exists():
        vec_env = VecNormalize.load(str(norm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    try:
        model = SAC.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}. Please train first.")
        return

    print(f"Running {num_episodes} evaluation episodes...")

    # Data collection
    all_trajectories = []       # list of (x, y) arrays
    collision_points = []       # (x, y) where collision occurred
    episode_lengths = []
    
    # Environment geometry collection for density heatmap
    all_obstacles = []
    all_goals = []
    
    # Simple manual approximation of reward breakdown (since environment groups them)
    # We will log successful vs collision terminations
    outcomes = {"Goal Reached": 0, "Collision": 0, "Timeout": 0}

    for ep in range(num_episodes):
        obs = vec_env.reset()
        
        # Save this episode's geometry
        all_obstacles.extend(raw_env.obstacle_rects.copy())
        all_goals.append((raw_env.goal_pos.copy(), raw_env.goal_radius))
        
        traj = []
        ep_len = 0
        done = False
        
        while not done:
            # We must access raw_env to get the true un-normalized position
            pos = raw_env.robot_pos.copy()
            traj.append(pos)
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env.step(action)
            ep_len += 1
            
            if dones[0]:
                done = True
                r = reward[0]
                if r > 50:
                    outcomes["Goal Reached"] += 1
                elif r < -5:
                    outcomes["Collision"] += 1
                    collision_points.append(raw_env.robot_pos.copy())
                else:
                    outcomes["Timeout"] += 1

        all_trajectories.append(np.array(traj))
        episode_lengths.append(ep_len)
        if (ep + 1) % 10 == 0:
            print(f"  ... {ep + 1}/{num_episodes} episodes completed")

    print("\n--- Evaluation Results ---")
    print(f"Success Rate:  {outcomes['Goal Reached'] / num_episodes * 100:.1f}%")
    print(f"Collision Rate: {outcomes['Collision'] / num_episodes * 100:.1f}%")
    print(f"Timeout Rate:  {outcomes['Timeout'] / num_episodes * 100:.1f}%")
    print(f"Avg Episode Length: {np.mean(episode_lengths):.1f} steps")

    # =========================================================================
    # Plotting
    # =========================================================================
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Agent Analytics Dashboard ({num_episodes} Episodes)", fontsize=18, fontweight="bold", y=0.96)

    # 1. Trajectory Heatmap (Main plot)
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
    ax1.set_title("Navigation Trajectories & Environments", fontsize=14)
    ax1.set_xlim(0, raw_env.world_w)
    ax1.set_ylim(raw_env.world_h, 0) # Invert Y to match Pygame coordinates
    
    # Plot obstacle density heatmap
    # Calculate a suitable alpha based on episode count
    alpha_val = max(0.02, min(0.3, 5.0 / num_episodes))
    
    for (ox, oy, ow, oh) in all_obstacles:
        rect = plt.Rectangle((ox, oy), ow, oh, facecolor="#e74c3c", alpha=alpha_val, edgecolor="none")
        ax1.add_patch(rect)
        
    for g_pos, g_rad in all_goals:
        circle = plt.Circle((g_pos[0], g_pos[1]), g_rad, color="#2ecc71", alpha=alpha_val, edgecolor="none")
        ax1.add_patch(circle)

    # Plot trajectories
    for traj in all_trajectories:
        ax1.plot(traj[:, 0], traj[:, 1], color="#3498db", alpha=0.1, linewidth=2)
        
    # Plot collisions
    if collision_points:
        cx, cy = zip(*collision_points)
        ax1.scatter(cx, cy, color="red", marker="x", s=100, label="Collisions")
        ax1.legend()

    # 2. Outcomes Pie Chart
    ax2 = plt.subplot2grid((2, 3), (0, 2))
    ax2.set_title("Episode Outcomes", fontsize=14)
    labels = list(outcomes.keys())
    sizes = list(outcomes.values())
    colors = ["#2ecc71", "#e74c3c", "#f39c12"]
    ax2.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90, 
            wedgeprops={"edgecolor": "black", "linewidth": 1, "antialiased": True})

    # 3. Episode Length Histogram
    ax3 = plt.subplot2grid((2, 3), (1, 2))
    ax3.set_title("Episode Length Distribution", fontsize=14)
    ax3.hist(episode_lengths, bins=20, color="#9b59b6", edgecolor="black")
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Frequency")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("analytics_dashboard.png", dpi=150)
    print("\nDashboard saved to 'analytics_dashboard.png'. Displaying now...")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="robot_nav_model.zip")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    
    generate_analytics(args.model, args.config, args.episodes)
