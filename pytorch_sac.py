import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, max_size=1000000, device='cuda'):
        self.device = device
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.obs = torch.zeros((max_size, obs_dim), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((max_size, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((max_size, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        self.dones = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        
    def add(self, obs, next_obs, action, reward, done):
        batch_size = obs.shape[0]
        end_idx = self.ptr + batch_size
        
        if end_idx <= self.max_size:
            self.obs[self.ptr:end_idx] = obs
            self.next_obs[self.ptr:end_idx] = next_obs
            self.actions[self.ptr:end_idx] = action
            self.rewards[self.ptr:end_idx] = reward.unsqueeze(1)
            self.dones[self.ptr:end_idx] = done.unsqueeze(1)
            self.ptr = end_idx % self.max_size
            self.size = min(self.size + batch_size, self.max_size)
        else:
            overflow = end_idx - self.max_size
            first_part = self.max_size - self.ptr
            self.obs[self.ptr:] = obs[:first_part]
            self.next_obs[self.ptr:] = next_obs[:first_part]
            self.actions[self.ptr:] = action[:first_part]
            self.rewards[self.ptr:] = reward[:first_part].unsqueeze(1)
            self.dones[self.ptr:] = done[:first_part].unsqueeze(1)
            
            self.obs[:overflow] = obs[first_part:]
            self.next_obs[:overflow] = next_obs[first_part:]
            self.actions[:overflow] = action[first_part:]
            self.rewards[:overflow] = reward[first_part:].unsqueeze(1)
            self.dones[:overflow] = done[first_part:].unsqueeze(1)
            self.ptr = overflow
            self.size = self.max_size
            
    def sample(self, batch_size):
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.obs[idxs],
            self.next_obs[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.dones[idxs]
        )

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
    def forward(self, obs):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        # Reparameterization trick
        dist = torch.distributions.Normal(mu, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        
        # Enforcing Action Bound
        log_prob = dist.log_prob(x_t)
        # log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Q1 architecture
        self.q1_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Q2 architecture
        self.q2_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, obs, action):
        xu = torch.cat([obs, action], 1)
        return self.q1_net(xu), self.q2_net(xu)

class SAC:
    def __init__(self, obs_dim, action_dim, device='cuda',
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, target_entropy=None):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = Critic(obs_dim, action_dim).to(device)
        self.critic_target = Critic(obs_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Auto entropy tuning
        if target_entropy is None:
            target_entropy = -action_dim
        self.target_entropy = target_entropy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.alpha = self.log_alpha.exp().item()
        
    def select_action(self, obs, evaluate=False):
        with torch.no_grad():
            if evaluate:
                # Deterministic
                net_out = self.actor.net(obs)
                mu = self.actor.mu_layer(net_out)
                return torch.tanh(mu)
            else:
                action, _ = self.actor(obs)
                return action
                
    def update(self, buffer, batch_size):
        obs, next_obs, action, reward, done = buffer.sample(batch_size)
        
        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_obs)
            target_q1, target_q2 = self.critic_target(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q
            
        current_q1, current_q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        pi, log_pi = self.actor(obs)
        q1_pi, q2_pi = self.critic(obs, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        
        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp().item()
        
        soft_update(self.critic_target, self.critic, self.tau)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha,
            'alpha_loss': alpha_loss.item()
        }

    def save(self, filepath):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'log_alpha': self.log_alpha,
        }, filepath)
        
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = self.log_alpha.exp().item()
        self.critic_target.load_state_dict(self.critic.state_dict())
