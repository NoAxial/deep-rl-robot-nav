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

import matplotlib.pyplot as plt
import numpy as np
import stable_baselines3
try:
    import sb3_contrib
except ImportError:
    sb3_contrib = None
from matplotlib.collections import PatchCollection
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack

from config import load_config
from robot_env import RobotNavEnv
from model_utils import normalization_path_for, resolve_model_path


def generate_analytics(model_path: str, config_path: str | None = None, num_episodes: int = 100):
    cfg = load_config(config_path)

    # Disable dynamic obstacles during analytics to get a clear heatmap of static navigation
    cfg["world"]["dynamic_obstacles"] = False

    try:
        resolved_model = resolve_model_path(model_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}. Please train first.")
        return

    raw_env = RobotNavEnv(config=cfg)
    vec_env = DummyVecEnv([lambda: raw_env])
    vec_env = VecFrameStack(vec_env, n_stack=4)

    norm_path = normalization_path_for(resolved_model)
    if norm_path.is_file():
        vec_env = VecNormalize.load(str(norm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    elif cfg["training"]["normalize_obs"]:
        print(f"Warning: normalization statistics not found beside {resolved_model}")

    algo_name = cfg.get("training", {}).get("algorithm", "SAC")
    if algo_name in ["RecurrentPPO", "TQC"] and sb3_contrib:
        AlgoClass = getattr(sb3_contrib, algo_name)
    else:
        AlgoClass = getattr(stable_baselines3, algo_name)

    model = AlgoClass.load(str(resolved_model))
    print(f"Loaded model: {resolved_model}")

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
    ep_data = []

    obs = vec_env.reset()

    for ep in range(num_episodes):
        # Save this episode's geometry
        all_obstacles.extend(raw_env.obstacle_rects.copy())
        all_goals.append((raw_env.goal_pos.copy(), raw_env.goal_radius))
        
        traj = []
        ep_len = 0
        ep_reward = 0.0
        n_obstacles = len(raw_env.obstacles)
        done = False
        
        while not done:
            # We must access raw_env to get the true un-normalized position
            pos = raw_env.robot_pos.copy()
            traj.append(pos)
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env.step(action)
            ep_reward += reward[0]
            ep_len += 1
            
            if dones[0]:
                done = True
                info = infos[0]
                terminal_info = info.get('terminal_info', info)
                if terminal_info.get("is_success", False):
                    outcomes["Goal Reached"] += 1
                    outcome_str = "Goal Reached"
                elif terminal_info.get("is_collision", False):
                    outcomes["Collision"] += 1
                    collision_points.append(traj[-1].copy())
                    outcome_str = "Collision"
                else:
                    outcomes["Timeout"] += 1
                    outcome_str = "Timeout"

                ep_data.append({
                    "n_obstacles": n_obstacles,
                    "ep_reward": ep_reward,
                    "outcome": outcome_str
                })

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
    
    patches_list = []
    for (ox, oy, ow, oh) in all_obstacles:
        patches_list.append(plt.Rectangle((ox, oy), ow, oh))
    
    collection = PatchCollection(patches_list, facecolor="#e74c3c", alpha=alpha_val, edgecolor="none")
    ax1.add_collection(collection)
        
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
    print("\nDashboard saved to 'analytics_dashboard.png'.")

    # 4. Success by Density Stacked Bar Chart
    import pandas as pd
    df = pd.DataFrame(ep_data)
    if not df.empty:
        density_outcomes = df.groupby(['n_obstacles', 'outcome']).size().unstack(fill_value=0)
        for col in ["Goal Reached", "Collision", "Timeout"]:
            if col not in density_outcomes.columns:
                density_outcomes[col] = 0
        density_outcomes = density_outcomes[["Goal Reached", "Collision", "Timeout"]]
        
        fig2, ax_dens = plt.subplots(figsize=(8, 6))
        density_outcomes.plot(kind='bar', stacked=True, color=colors, ax=ax_dens, edgecolor='black')
        ax_dens.set_title("Episode Outcomes by Obstacle Density")
        ax_dens.set_xlabel("Number of Obstacles")
        ax_dens.set_ylabel("Number of Episodes")
        plt.tight_layout()
        plt.savefig("success_by_density.png", dpi=150)
        plt.close(fig2)
        print("Success by density chart saved to 'success_by_density.png'.")

    vec_env.close()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="robot_nav_model.zip")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    
    generate_analytics(args.model, args.config, args.episodes)
