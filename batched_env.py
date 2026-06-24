import torch
import numpy as np

class BatchedRobotEnv:
    def __init__(self, config, num_envs=4096, device='cuda'):
        self.config = config
        self.num_envs = num_envs
        self.device = torch.device(device)
        
        w_cfg = config["world"]
        r_cfg = config["robot"]
        g_cfg = config["goal"]
        rw_cfg = config["rewards"]
        ep_cfg = config["episode"]

        self.world_w = float(w_cfg["width"])
        self.world_h = float(w_cfg["height"])
        self.obs_min_n = int(w_cfg["num_obstacles_min"])
        self.obs_max_n = int(w_cfg["num_obstacles_max"])
        self.obs_w_range = (float(w_cfg["obstacle_width_min"]), float(w_cfg["obstacle_width_max"]))
        self.obs_h_range = (float(w_cfg["obstacle_height_min"]), float(w_cfg["obstacle_height_max"]))

        self.robot_radius = float(r_cfg["radius"])
        self.num_sensors = int(r_cfg["num_sensors"])
        self.sensor_range = float(r_cfg["sensor_range"])
        self.max_linear_speed = float(r_cfg["max_linear_speed"])
        self.max_angular_speed = float(r_cfg["max_angular_speed"])

        self.goal_radius = float(g_cfg["radius"])
        self.max_steps = int(ep_cfg["max_steps"])
        self.dt = 0.1

        # Rewards
        self.rw_goal = float(rw_cfg["goal"])
        self.rw_collision = float(rw_cfg["collision"])
        self.rw_step = float(rw_cfg["step_penalty"])
        self.rw_vel = float(rw_cfg.get("velocity_reward", 0.5))
        self.rw_dist = float(rw_cfg.get("distance_shaping", 0.0))
        self.rw_heading = float(rw_cfg.get("heading_bonus", 0.0))
        self.rw_prox = float(rw_cfg["proximity_penalty"])
        self.rw_prox_thresh = float(rw_cfg["proximity_threshold"])
        self.rw_speed = float(rw_cfg["speed_bonus"])
        self.rw_spin = float(rw_cfg["spin_penalty"])
        self.rw_jerk = float(rw_cfg.get("jerk_penalty", 0.1))
        self.reverse_penalty = float(rw_cfg.get("reverse_penalty", 0.5))
        self.rw_target_acquired = 0.0
        self.rw_target_visible = 0.0

        # Sensor Angles
        s_cfg = config.get("sensors", {})
        if s_cfg.get("lidar_distribution") == "non_uniform":
            front_ratio = float(s_cfg.get("front_arc_ratio", 0.5))
            front_deg = float(s_cfg.get("front_arc_degrees", 120))
            front_rad = np.radians(front_deg)
            n_front = int(self.num_sensors * front_ratio)
            n_back = self.num_sensors - n_front
            front_angles = np.linspace(-front_rad/2, front_rad/2, n_front, endpoint=False)
            back_angles = np.linspace(front_rad/2, 2*np.pi - front_rad/2, n_back, endpoint=False)
            ray_angles = np.concatenate([front_angles, back_angles])
        else:
            ray_angles = np.linspace(0, 2 * np.pi, self.num_sensors, endpoint=False)
        self.ray_angles = torch.tensor(ray_angles, dtype=torch.float32, device=self.device)

        # State Tensors
        self.robot_pos = torch.zeros((num_envs, 2), dtype=torch.float32, device=self.device)
        self.robot_angle = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.goal_pos = torch.zeros((num_envs, 2), dtype=torch.float32, device=self.device)
        self.obstacles = torch.zeros((num_envs, self.obs_max_n, 4), dtype=torch.float32, device=self.device)
        self.obstacle_velocities = torch.zeros((num_envs, self.obs_max_n, 2), dtype=torch.float32, device=self.device)
        self.num_obstacles = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        
        self.difficulty = 0.0

        self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_action = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.prev_dist_to_goal = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.v = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.omega = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        
        self.obs_dim_single = self.num_sensors + 7
        self.frame_stack = 3
        self.obs_buffer = torch.zeros((self.num_envs, self.frame_stack, self.obs_dim_single), dtype=torch.float32, device=self.device)
        self.obs_dim = self.obs_dim_single * self.frame_stack
        self.action_dim = 2

    def set_difficulty(self, difficulty):
        self.difficulty = max(0.0, min(1.0, float(difficulty)))

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        n = len(env_ids)
        if n == 0:
            return

        self.step_count[env_ids] = 0
        self.prev_action[env_ids] = 0.0
        self.v[env_ids] = 0.0
        self.omega[env_ids] = 0.0

        # Robot spawns on the left
        margin = 60.0
        self.robot_pos[env_ids, 0] = torch.rand(n, device=self.device) * (self.world_w * 0.30 - margin) + margin
        self.robot_pos[env_ids, 1] = torch.rand(n, device=self.device) * (self.world_h - 2*margin) + margin
        self.robot_angle[env_ids] = torch.rand(n, device=self.device) * 2 * np.pi - np.pi

        # Goal spawns at a distance that scales with difficulty
        min_dist_x = 100.0
        max_dist_x = self.world_w * 0.8 - margin
        target_dist_x = min_dist_x + (max_dist_x - min_dist_x) * self.difficulty
        
        # Goal x position is somewhat ahead of the robot
        self.goal_pos[env_ids, 0] = self.robot_pos[env_ids, 0] + torch.rand(n, device=self.device) * 50.0 + target_dist_x
        self.goal_pos[env_ids, 0] = torch.clamp(self.goal_pos[env_ids, 0], max=self.world_w - margin)
        self.goal_pos[env_ids, 1] = torch.rand(n, device=self.device) * (self.world_h - 2*margin) + margin
        self.prev_dist_to_goal[env_ids] = torch.norm(self.goal_pos[env_ids] - self.robot_pos[env_ids], dim=-1)

        # Generate obstacles (count scales with difficulty)
        current_obs_max = int(self.obs_max_n * self.difficulty)
        current_obs_min = int(self.obs_min_n * self.difficulty)
        
        if current_obs_max == 0:
            num_obs = torch.zeros(n, dtype=torch.long, device=self.device)
        else:
            num_obs = torch.randint(current_obs_min, current_obs_max + 1, (n,), device=self.device)
            
        self.num_obstacles[env_ids] = num_obs

        for i in range(self.obs_max_n):
            mask = i < num_obs
            if not mask.any(): continue
            valid_envs = env_ids[mask]
            k = len(valid_envs)
            w = torch.rand(k, device=self.device) * (self.obs_w_range[1] - self.obs_w_range[0]) + self.obs_w_range[0]
            h = torch.rand(k, device=self.device) * (self.obs_h_range[1] - self.obs_h_range[0]) + self.obs_h_range[0]
            
            max_x = torch.clamp(torch.tensor(self.world_w - 20.0, device=self.device) - w, min=20.0)
            max_y = torch.clamp(torch.tensor(self.world_h - 20.0, device=self.device) - h, min=20.0)
            
            x = torch.rand(k, device=self.device) * (max_x - 20.0) + 20.0
            y = torch.rand(k, device=self.device) * (max_y - 20.0) + 20.0

            # Simplified: In a full tensor implementation, we skip complex overlap checks 
            # for speed, relying on random scatter. If strict overlap checking is needed,
            # we do rejection sampling. For now, random scatter is sufficient for RL.
            self.obstacles[valid_envs, i, 0] = x
            self.obstacles[valid_envs, i, 1] = y
            self.obstacles[valid_envs, i, 2] = w
            self.obstacles[valid_envs, i, 3] = h
            
            # Dynamic random velocities: scale with difficulty
            vx = ((torch.rand(k, device=self.device) * 60.0) - 30.0) * self.difficulty
            vy = ((torch.rand(k, device=self.device) * 60.0) - 30.0) * self.difficulty
            self.obstacle_velocities[valid_envs, i, 0] = vx
            self.obstacle_velocities[valid_envs, i, 1] = vy

        # For inactive obstacles, place them far out of bounds
        for i in range(self.obs_max_n):
            mask = i >= num_obs
            if not mask.any(): continue
            valid_envs = env_ids[mask]
            self.obstacles[valid_envs, i, :] = -1000.0
            self.obstacle_velocities[valid_envs, i, :] = 0.0

        # Reset frame stacking buffer for these envs
        self.obs_buffer[env_ids] = 0.0
        obs_single = self._get_obs_single()[env_ids]
        for f in range(self.frame_stack):
            self.obs_buffer[env_ids, f] = obs_single

        return self.obs_buffer.view(self.num_envs, -1)

    def step(self, action):
        action = torch.clamp(action, -1.0, 1.0)
        self.step_count += 1
        
        target_v = action[:, 0] * self.max_linear_speed
        target_omega = action[:, 1] * self.max_angular_speed
        
        self.v = target_v
        self.omega = target_omega
        
        self.robot_angle += self.omega * self.dt
        self.robot_angle = (self.robot_angle + np.pi) % (2 * np.pi) - np.pi
        
        self.robot_pos[:, 0] += self.v * torch.cos(self.robot_angle) * self.dt
        self.robot_pos[:, 1] += self.v * torch.sin(self.robot_angle) * self.dt
        
        # Dynamic Obstacle Updates
        self.obstacles[..., 0] += self.obstacle_velocities[..., 0] * self.dt
        self.obstacles[..., 1] += self.obstacle_velocities[..., 1] * self.dt
        
        ox = self.obstacles[..., 0]
        oy = self.obstacles[..., 1]
        ow = self.obstacles[..., 2]
        oh = self.obstacles[..., 3]
        vx = self.obstacle_velocities[..., 0]
        vy = self.obstacle_velocities[..., 1]
        
        # Bounce obstacles off walls
        bounce_x = (ox <= 0) | (ox + ow >= self.world_w)
        bounce_y = (oy <= 0) | (oy + oh >= self.world_h)
        self.obstacle_velocities[..., 0] = torch.where(bounce_x, -vx, vx)
        self.obstacle_velocities[..., 1] = torch.where(bounce_y, -vy, vy)
        
        # Clamp obstacles to stay strictly inside the world
        max_x = self.world_w - ow
        max_y = self.world_h - oh
        self.obstacles[..., 0] = torch.maximum(torch.zeros_like(ox), torch.minimum(self.obstacles[..., 0], max_x))
        self.obstacles[..., 1] = torch.maximum(torch.zeros_like(oy), torch.minimum(self.obstacles[..., 1], max_y))
        
        dist_to_goal = torch.norm(self.goal_pos - self.robot_pos, dim=-1)
        angle_to_goal = torch.atan2(self.goal_pos[:, 1] - self.robot_pos[:, 1], self.goal_pos[:, 0] - self.robot_pos[:, 0])
        
        # Observation
        obs_single = self._get_obs_single()
        self.obs_buffer = torch.roll(self.obs_buffer, shifts=-1, dims=1)
        self.obs_buffer[:, -1, :] = obs_single
        obs = self.obs_buffer.view(self.num_envs, -1).clone()
        
        # Collisions
        x = self.robot_pos[:, 0]
        y = self.robot_pos[:, 1]
        r = self.robot_radius
        out_of_bounds = (x - r <= 0) | (x + r >= self.world_w) | (y - r <= 0) | (y + r >= self.world_h)
        
        # Re-fetch clamped obstacle positions for collision
        ox = self.obstacles[..., 0]
        oy = self.obstacles[..., 1]
        
        closest_x = torch.clamp(x.unsqueeze(1), ox, ox + ow)
        closest_y = torch.clamp(y.unsqueeze(1), oy, oy + oh)
        dx = x.unsqueeze(1) - closest_x
        dy = y.unsqueeze(1) - closest_y
        dist_sq = dx**2 + dy**2
        
        # Proximity Penalty (soft obstacle avoidance)
        min_dist_to_obs = torch.sqrt(dist_sq).min(dim=1)[0] - r
        is_close = min_dist_to_obs < (self.rw_prox_thresh * self.sensor_range)
        
        hit_obs = (dist_sq <= r**2).any(dim=1)
        
        collision = out_of_bounds | hit_obs
        goal_reached = dist_to_goal < (r + self.goal_radius)
        
        terminated = collision | goal_reached
        truncated = self.step_count >= self.max_steps
        done = terminated | truncated
        
        # Rewards (Pure Kinematic Smooth Driving)
        reward = torch.full((self.num_envs,), self.rw_step, device=self.device)
        reward[collision] += self.rw_collision
        
        time_bonus = torch.clamp(1.0 - self.step_count.float() / self.max_steps, min=0.0)
        reward[goal_reached] += self.rw_goal + self.rw_speed * time_bonus[goal_reached]
        
        not_done = ~done
        if not_done.any():
            nd = not_done
            
            # Distance shaping (heavily reward getting closer)
            dist_diff = self.prev_dist_to_goal[nd] - dist_to_goal[nd]
            reward[nd] += self.rw_dist * dist_diff
            
            # Heading bonus
            heading_err = self._angle_diff(self.robot_angle[nd], angle_to_goal[nd])
            norm_speed = torch.abs(self.v[nd]) / self.max_linear_speed
            reward[nd] += self.rw_heading * torch.cos(heading_err) * norm_speed
            
            # Action smoothing (penalize seizures)
            jerk = torch.norm(action[nd] - self.prev_action[nd], dim=-1)
            reward[nd] -= self.rw_jerk * jerk
            
            # Proximity penalty
            reward[nd] -= torch.where(is_close[nd], self.rw_prox * (1.0 - min_dist_to_obs[nd] / (self.rw_prox_thresh * self.sensor_range)), torch.zeros_like(min_dist_to_obs[nd]))
            
            # Spinning & reversing penalties
            reward[nd] -= self.rw_spin * torch.abs(self.omega[nd]) / self.max_angular_speed
            reward[nd] -= torch.where(self.v[nd] < 0, self.reverse_penalty * torch.abs(self.v[nd]) / self.max_linear_speed, torch.zeros_like(self.v[nd]))

        self.prev_action = action.clone()
        self.prev_dist_to_goal = dist_to_goal.clone()
        
        # Auto-reset
        if done.any():
            self.reset(torch.nonzero(done).squeeze(-1))
            obs[done] = self.obs_buffer[done].view(torch.sum(done).item(), -1).clone()
            
        info = {
            'goals': goal_reached.sum().item(),
            'dones': done.sum().item()
        }
            
        return obs, reward, done, info

    def _cast_ray_from(self, start_pos, angle):
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        
        dists = torch.full((self.num_envs,), self.sensor_range, dtype=torch.float32, device=self.device)
        rx = start_pos[:, 0]
        ry = start_pos[:, 1]
        
        inv_cos = torch.where(cos_a == 0, 1e-8, cos_a).reciprocal()
        inv_sin = torch.where(sin_a == 0, 1e-8, sin_a).reciprocal()
        
        tx1 = (0.0 - rx) * inv_cos
        tx2 = (self.world_w - rx) * inv_cos
        ty1 = (0.0 - ry) * inv_sin
        ty2 = (self.world_h - ry) * inv_sin
        
        tmin_x = torch.minimum(tx1, tx2)
        tmax_x = torch.maximum(tx1, tx2)
        tmin_y = torch.minimum(ty1, ty2)
        tmax_y = torch.maximum(ty1, ty2)
        
        tmin = torch.maximum(tmin_x, tmin_y)
        tmax = torch.minimum(tmax_x, tmax_y)
        
        valid = (tmax >= tmin) & (tmin >= 0)
        dists = torch.where(valid, torch.minimum(dists, tmin), dists)
        
        ox = self.obstacles[..., 0]
        oy = self.obstacles[..., 1]
        w = self.obstacles[..., 2]
        h = self.obstacles[..., 3]
        
        inv_cos_exp = inv_cos.unsqueeze(1)
        inv_sin_exp = inv_sin.unsqueeze(1)
        rx_exp = rx.unsqueeze(1)
        ry_exp = ry.unsqueeze(1)
        
        tx1 = (ox - rx_exp) * inv_cos_exp
        tx2 = (ox + w - rx_exp) * inv_cos_exp
        ty1 = (oy - ry_exp) * inv_sin_exp
        ty2 = (oy + h - ry_exp) * inv_sin_exp
        
        tmin_x = torch.minimum(tx1, tx2)
        tmax_x = torch.maximum(tx1, tx2)
        tmin_y = torch.minimum(ty1, ty2)
        tmax_y = torch.maximum(ty1, ty2)
        
        tmin = torch.maximum(tmin_x, tmin_y)
        tmax = torch.minimum(tmax_x, tmax_y)
        
        valid = (tmax >= tmin) & (tmin >= 0)
        tmin[~valid] = float('inf')
        
        min_obs_dist, _ = tmin.min(dim=1)
        dists = torch.minimum(dists, min_obs_dist)
        
        return dists

    def _get_obs_single(self):
        angles = self.robot_angle.unsqueeze(1) + self.ray_angles.unsqueeze(0)
        
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        
        dists = torch.full((self.num_envs, self.num_sensors), self.sensor_range, dtype=torch.float32, device=self.device)
        rx = self.robot_pos[:, 0].unsqueeze(1)
        ry = self.robot_pos[:, 1].unsqueeze(1)
        
        inv_cos = torch.where(cos_a == 0, 1e-8, cos_a).reciprocal()
        inv_sin = torch.where(sin_a == 0, 1e-8, sin_a).reciprocal()
        
        tx1 = (0.0 - rx) * inv_cos
        tx2 = (self.world_w - rx) * inv_cos
        ty1 = (0.0 - ry) * inv_sin
        ty2 = (self.world_h - ry) * inv_sin
        
        tmin_x = torch.minimum(tx1, tx2)
        tmax_x = torch.maximum(tx1, tx2)
        tmin_y = torch.minimum(ty1, ty2)
        tmax_y = torch.maximum(ty1, ty2)
        
        tmin = torch.maximum(tmin_x, tmin_y)
        tmax = torch.minimum(tmax_x, tmax_y)
        
        valid = (tmax >= tmin) & (tmin >= 0)
        dists = torch.where(valid, torch.minimum(dists, tmin), dists)
        
        ox = self.obstacles[..., 0].unsqueeze(1)
        oy = self.obstacles[..., 1].unsqueeze(1)
        w = self.obstacles[..., 2].unsqueeze(1)
        h = self.obstacles[..., 3].unsqueeze(1)
        
        inv_cos_exp = inv_cos.unsqueeze(2)
        inv_sin_exp = inv_sin.unsqueeze(2)
        rx_exp = rx.unsqueeze(2)
        ry_exp = ry.unsqueeze(2)
        
        tx1 = (ox - rx_exp) * inv_cos_exp
        tx2 = (ox + w - rx_exp) * inv_cos_exp
        ty1 = (oy - ry_exp) * inv_sin_exp
        ty2 = (oy + h - ry_exp) * inv_sin_exp
        
        tmin_x = torch.minimum(tx1, tx2)
        tmax_x = torch.maximum(tx1, tx2)
        tmin_y = torch.minimum(ty1, ty2)
        tmax_y = torch.maximum(ty1, ty2)
        
        tmin = torch.maximum(tmin_x, tmin_y)
        tmax = torch.minimum(tmax_x, tmax_y)
        
        valid = (tmax >= tmin) & (tmin >= 0)
        tmin[~valid] = float('inf')
        
        min_obs_dist, _ = tmin.min(dim=2)
        dists = torch.minimum(dists, min_obs_dist)
        
        lidar = dists / self.sensor_range
        
        delta = self.goal_pos - self.robot_pos
        cos_r = torch.cos(self.robot_angle)
        sin_r = torch.sin(self.robot_angle)
        
        # Transform world delta to robot's local frame (requires inverse rotation matrix)
        target_dx = delta[:, 0] * cos_r + delta[:, 1] * sin_r
        target_dy = -delta[:, 0] * sin_r + delta[:, 1] * cos_r
        diag = np.sqrt(self.world_w**2 + self.world_h**2)
        
        extras = torch.stack([
            target_dx / diag,
            target_dy / diag,
            self.prev_action[:, 0],
            self.prev_action[:, 1],
            self.step_count.float() / self.max_steps,
            self.v / self.max_linear_speed,
            self.omega / self.max_angular_speed
        ], dim=1)
        
        obs = torch.cat([lidar, extras], dim=1)
        return obs

    @staticmethod
    def _angle_diff(a1, a2):
        d = (a2 - a1) % (2.0 * np.pi)
        d = torch.where(d > np.pi, d - 2.0 * np.pi, d)
        return d
