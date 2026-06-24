import torch
import time
import os
from config import load_config
from batched_env import BatchedRobotEnv
from pytorch_sac import SAC, ReplayBuffer

def train():
    config_file = "config.yaml"
    config = load_config(config_file)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_envs = 4096
    env = BatchedRobotEnv(config, num_envs=num_envs, device=device)
    
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    
    sac = SAC(obs_dim, action_dim, device=device)
    buffer = ReplayBuffer(obs_dim, action_dim, max_size=1000000, device=device)
    
    total_timesteps = 100_000_000
    batch_size = 4096
    updates_per_step = 4 # How many gradient updates to do per env.step()
    
    obs = env.reset()
    
    global_step = 0
    start_time = time.time()
    
    total_goals = 0
    total_dones = 0
    
    print(f"Starting training for {total_timesteps} timesteps...")
    
    while global_step < total_timesteps:
        # Curriculum Learning: Scale difficulty over the first 20,000,000 steps
        difficulty = min(1.0, global_step / 20_000_000.0)
        env.set_difficulty(difficulty)
        
        # 1. Collect experience
        # Exploration noise is handled by the stochastic policy in SAC
        action = sac.select_action(obs, evaluate=False)
        
        next_obs, reward, done, info = env.step(action)
        
        total_goals += info.get('goals', 0)
        total_dones += info.get('dones', 0)
        
        buffer.add(obs, next_obs, action, reward, done)
        obs = next_obs
        
        global_step += num_envs
        
        # 2. Update network
        if buffer.size >= batch_size:
            for _ in range(updates_per_step):
                metrics = sac.update(buffer, batch_size)
                
            if (global_step // num_envs) % 10 == 0:
                elapsed = time.time() - start_time
                fps = global_step / elapsed
                success_rate = (total_goals / max(1, total_dones)) * 100
                print(f"Step: {global_step}/{total_timesteps} | FPS: {fps:.0f} | "
                      f"Succ: {success_rate:.1f}% | "
                      f"Critic Loss: {metrics['critic_loss']:.3f} | "
                      f"Actor Loss: {metrics['actor_loss']:.3f} | "
                      f"Alpha: {metrics['alpha']:.3f}")
                      
    # Save final model
    os.makedirs("models", exist_ok=True)
    save_path = "models/sac_gpu_final.pt"
    sac.save(save_path)
    print(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    train()
