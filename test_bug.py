import torch
from pytorch_sac import SAC, ReplayBuffer
from batched_env import BatchedRobotEnv
from config import load_config

config = load_config('config.yaml')
env = BatchedRobotEnv(config, num_envs=256, device='cuda' if torch.cuda.is_available() else 'cpu')
obs = env.reset()
sac = SAC(env.obs_dim, env.action_dim, device=env.device)
buffer = ReplayBuffer(env.obs_dim, env.action_dim, max_size=10000, device=env.device)

for i in range(200):
    action = sac.select_action(obs, evaluate=False)
    next_obs, reward, done, _ = env.step(action)
    buffer.add(obs, next_obs, action, reward, done)
    obs = next_obs
    if buffer.size >= 256:
        metrics = sac.update(buffer, 256)
        if i % 50 == 0:
            print(f"Step {i} | Reward: {reward.mean().item():.2f} | Critic: {metrics['critic_loss']:.2f} | Actor: {metrics['actor_loss']:.2f} | Dones: {done.sum().item()}")
