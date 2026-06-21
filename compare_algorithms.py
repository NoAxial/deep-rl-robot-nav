"""
compare_algorithms.py -- Train and compare multiple RL algorithms.

Trains PPO and A2C on the same RobotNavEnv for a short number of timesteps
and plots their learning curves.

Usage:
  py compare_algorithms.py --timesteps 50000
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecFrameStack

from config import load_config
from robot_env import RobotNavEnv


class EvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=2000, n_eval_episodes=10):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_timesteps = []
        self.eval_rewards = []

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            rewards = []
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                ep_r = 0
                while not done:
                    # deterministic=True for evaluation
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, r, dones, _ = self.eval_env.step(action)
                    ep_r += r[0]
                    if dones[0]:
                        done = True
                rewards.append(ep_r)
            
            mean_r = np.mean(rewards)
            self.eval_timesteps.append(self.num_timesteps)
            self.eval_rewards.append(mean_r)
            print(f"[{self.model.__class__.__name__}] Step {self.num_timesteps}: Mean Reward = {mean_r:.2f}")
        return True


def compare(timesteps: int, config_path: str | None = None):
    cfg = load_config(config_path)

    # Disable dynamic obstacles during comparison for stability
    cfg["world"]["dynamic_obstacles"] = False

    def make_env():
        return RobotNavEnv(config=cfg)

    algorithm_classes = {
        "SAC": lambda env: SAC("MlpPolicy", env, verbose=0, device="cpu", buffer_size=10000, learning_starts=500),
        "PPO": lambda env: PPO("MlpPolicy", env, verbose=0, device="cpu", n_steps=1024)
    }

    results = {}

    for name, make_model in algorithm_classes.items():
        print(f"\n==============================================")
        print(f" Training {name}")
        print(f"==============================================")
        
        # Fresh env and model for each algorithm
        vec_env = DummyVecEnv([make_env for _ in range(4)])
        vec_env = VecFrameStack(vec_env, n_stack=4)
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        
        model = make_model(vec_env)
        
        eval_env = DummyVecEnv([make_env])
        eval_env = VecFrameStack(eval_env, n_stack=4)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training_env=vec_env)
        
        callback = EvalCallback(eval_env, eval_freq=max(1024, timesteps // 20), n_eval_episodes=5)
        
        start_t = time.time()
        model.learn(total_timesteps=timesteps, callback=callback)
        time_taken = time.time() - start_t
        
        results[name] = {
            "timesteps": callback.eval_timesteps,
            "rewards": callback.eval_rewards,
            "time": time_taken
        }

    # Plotting
    plt.style.use("dark_background")
    plt.figure(figsize=(10, 6))
    
    colors = {"SAC": "#e74c3c", "PPO": "#3498db"}
    
    for name, data in results.items():
        plt.plot(data["timesteps"], data["rewards"], marker='o', linewidth=2, 
                 label=f"{name} ({data['time']:.1f}s)", color=colors[name])

    plt.title(f"Algorithm Comparison ({timesteps} steps)")
    plt.xlabel("Timesteps")
    plt.ylabel("Evaluation Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("algorithm_comparison.png", dpi=150)
    print("\nComparison complete. Chart saved to 'algorithm_comparison.png'.")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    
    compare(args.timesteps, args.config)
