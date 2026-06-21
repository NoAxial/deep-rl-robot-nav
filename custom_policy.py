import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor for SAC.
    Splits the 1D observation (which contains stacked frames) into:
      1. LIDAR features (processed through a 1D pathway)
      2. State features (processed through an MLP)
    Then concatenates and merges them.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, num_sensors: int = 32, n_stack: int = 4):
        super().__init__(observation_space, features_dim)
        
        self.num_sensors = num_sensors
        self.n_stack = n_stack
        self.obs_dim_per_frame = num_sensors + 8 # 32 sensors + 8 state features
        
        # Verify observation shape matches expected stacked shape
        import math
        expected_dim = self.obs_dim_per_frame * self.n_stack
        actual_dim = math.prod(observation_space.shape)
        assert actual_dim == expected_dim, f"Expected obs shape {expected_dim}, got {observation_space.shape}"

        # LIDAR Pathway: We have (n_stack * num_sensors) features.
        # LIDAR Pathway: We have (n_stack * num_sensors) features.
        # We use a very lightweight 1D CNN over the spatial LIDAR dimension, treating n_stack as channels.
        # This allows the network to learn temporal motion (velocity) of obstacles from the frame stack!
        self.lidar_cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_stack, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate CNN output size
        # Input to conv: [batch, n_stack, 32]
        # After Conv 1 (stride 2): [batch, 16, 16]
        # After Conv 2 (stride 2): [batch, 32, 8]
        # Flattened: 32 * 8 = 256
        cnn_out_dim = 32 * (num_sensors // 4)
        
        # State Pathway: n_stack * 8 features
        state_in_dim = n_stack * 8
        self.state_mlp = nn.Sequential(
            nn.Linear(state_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Merged Pathway
        self.merged_mlp = nn.Sequential(
            nn.Linear(cnn_out_dim + 64, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: [batch_size, n_stack * obs_dim_per_frame]
        batch_size = observations.shape[0]
        
        # Reshape to [batch_size, n_stack, obs_dim_per_frame]
        obs_reshaped = observations.reshape(batch_size, self.n_stack, self.obs_dim_per_frame)
        
        # Split LIDAR and State
        # LIDAR: first `num_sensors` elements
        lidar_obs = obs_reshaped[:, :, :self.num_sensors] # Shape: [batch_size, n_stack, num_sensors]
        
        # State: remaining elements
        state_obs = obs_reshaped[:, :, self.num_sensors:] # Shape: [batch_size, n_stack, 8]
        
        # Pass LIDAR through CNN
        lidar_features = self.lidar_cnn(lidar_obs)
        
        # Pass State through MLP (flatten first)
        state_features = self.state_mlp(state_obs.reshape(batch_size, -1))
        
        # Concatenate
        combined = torch.cat((lidar_features, state_features), dim=1)
        
        # Final merge
        return self.merged_mlp(combined)
