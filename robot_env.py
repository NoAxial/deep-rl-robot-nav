"""
robot_env.py -- Custom Gymnasium Environment for 2D Robot Navigation.

A circular robot navigates a continuous 2D arena, using simulated LIDAR
sensors to detect walls and rectangular obstacles, and must reach a goal
location while avoiding collisions.

State  (15-D by default):
    0..N-1  normalised LIDAR readings
    N, N+1  relative goal [dx, dy] in robot local frame
    N+2     normalised distance to goal
    N+3     angle to goal (normalised to [-1, 1])
    N+4     previous action (normalised)
    N+5     normalised step count
    N+6     min LIDAR reading

Actions (Discrete 3):
    0 = Move Forward, 1 = Turn Left, 2 = Turn Right

Rewards (configurable):
    +goal          reaching goal (+ speed bonus)
    -collision     hitting wall / obstacle
    -step_penalty  every step
    +distance      shaping for approaching goal
    +heading       facing the goal
    -proximity     being too close to obstacles
    -spin          consecutive turn actions
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from shapely.geometry import LineString, Point
from shapely.geometry import box as shapely_box

from config import DEFAULT_CONFIG


class RobotNavEnv(gym.Env):
    """2-D robot navigation with LIDAR sensing (config-driven)."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: dict | None = None, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode

        # ---- Load config (fall back to defaults) ----
        cfg = config if config is not None else DEFAULT_CONFIG
        self.config = cfg
        w_cfg = cfg["world"]
        r_cfg = cfg["robot"]
        g_cfg = cfg["goal"]
        rw_cfg = cfg["rewards"]
        ep_cfg = cfg["episode"]

        # World
        self.world_w: int = int(w_cfg["width"])
        self.world_h: int = int(w_cfg["height"])
        self.obs_min_n: int = int(w_cfg["num_obstacles_min"])
        self.obs_max_n: int = int(w_cfg["num_obstacles_max"])
        self.obs_w_range = (float(w_cfg["obstacle_width_min"]), float(w_cfg["obstacle_width_max"]))
        self.obs_h_range = (float(w_cfg["obstacle_height_min"]), float(w_cfg["obstacle_height_max"]))
        self.dynamic_obstacles: bool = bool(w_cfg.get("dynamic_obstacles", False))
        self.dynamic_speed_range = (float(w_cfg.get("dynamic_speed_min", 1.0)), float(w_cfg.get("dynamic_speed_max", 3.0)))

        # Robot
        self.robot_radius: float = float(r_cfg["radius"])
        self.base_step_size: float = float(r_cfg["step_size"])
        self.base_turn_angle: float = np.radians(float(r_cfg["turn_angle"]))
        self.step_size: float = self.base_step_size
        self.turn_angle: float = self.base_turn_angle
        self.num_sensors: int = int(r_cfg["num_sensors"])
        self.base_sensor_range: float = float(r_cfg["sensor_range"])
        self.sensor_range: float = self.base_sensor_range
        self.action_mode: str = str(r_cfg.get("action_mode", "basic"))

        # Goal
        self.goal_radius: float = float(g_cfg["radius"])

        # Rewards
        self.rw_goal: float = float(rw_cfg["goal"])
        self.rw_collision: float = float(rw_cfg["collision"])
        self.rw_step: float = float(rw_cfg["step_penalty"])
        self.rw_vel: float = float(rw_cfg.get("velocity_reward", 0.5))
        self.rw_clear: float = float(rw_cfg.get("clear_path_reward", 0.1))
        self.rw_dist: float = float(rw_cfg.get("distance_shaping", 0.0))
        self.rw_heading: float = float(rw_cfg.get("heading_bonus", 0.0))
        self.rw_prox: float = float(rw_cfg["proximity_penalty"])
        self.rw_prox_thresh: float = float(rw_cfg["proximity_threshold"])
        self.rw_speed: float = float(rw_cfg["speed_bonus"])
        self.rw_spin: float = float(rw_cfg["spin_penalty"])
        self.rw_jerk: float = float(rw_cfg.get("jerk_penalty", 0.1))

        # Episode
        self.max_steps: int = int(ep_cfg["max_steps"])

        # ---- Spaces ----
        # Obs: N_sensors + 8 extra features (added speed)
        obs_dim = self.num_sensors + 8
        low = np.zeros(obs_dim, dtype=np.float32)
        high = np.ones(obs_dim, dtype=np.float32)
        # Relative goal dx, dy, angle_to_goal, and speed can be negative
        low[self.num_sensors]     = -1.0   # local_dx
        low[self.num_sensors + 1] = -1.0   # local_dy
        low[self.num_sensors + 3] = -1.0   # angle_to_goal
        low[self.num_sensors + 7] = -1.0   # speed
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        if self.action_mode == "continuous":
            self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        elif self.action_mode == "car_continuous":
            self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        elif self.action_mode == "extended":
            self.action_space = spaces.Discrete(6)
        elif self.action_mode == "car":
            self.action_space = spaces.Discrete(5)
        else:
            self.action_space = spaces.Discrete(3)

        # ---- Internal state (set in reset) ----
        self.robot_pos: np.ndarray = np.zeros(2)
        self.robot_angle: float = 0.0
        self.goal_pos: np.ndarray = np.zeros(2)
        self.obstacles: list = []
        self.obstacle_rects: list = []       # (x, y, w, h) for drawing
        self._obstacle_boundaries: list = []
        self.obstacle_vels: list = []        # Dynamic velocities
        self.step_count: int = 0
        self.robot_speed: float = 0.0        # For car_continuous mode
        self._prev_goal_dist: float = 0.0
        self._prev_action = None
        self._smoothed_action = None
        self.lidar_distances: np.ndarray = np.zeros(self.num_sensors)

        # Pre-build wall segments (constant for a given world size)
        W, H = self.world_w, self.world_h
        self._walls = [
            LineString([(0, 0), (W, 0)]),
            LineString([(W, 0), (W, H)]),
            LineString([(W, H), (0, H)]),
            LineString([(0, H), (0, 0)]),
        ]

    # ==================================================================
    # Obstacle generation
    # ==================================================================
    def _generate_obstacles(self) -> None:
        self.obstacles.clear()
        self.obstacle_rects.clear()
        self._obstacle_boundaries.clear()
        self.obstacle_vels.clear()

        n = int(self.np_random.integers(self.obs_min_n, self.obs_max_n + 1))
        robot_buf = Point(self.robot_pos).buffer(self.robot_radius + 45)
        goal_buf = Point(self.goal_pos).buffer(self.goal_radius + 45)

        for _ in range(n):
            for _ in range(80):
                w = float(self.np_random.uniform(*self.obs_w_range))
                h = float(self.np_random.uniform(*self.obs_h_range))
                x = float(self.np_random.uniform(20, self.world_w - 20 - w))
                y = float(self.np_random.uniform(20, self.world_h - 20 - h))
                rect = shapely_box(x, y, x + w, y + h)
                if rect.intersects(robot_buf) or rect.intersects(goal_buf):
                    continue
                if any(rect.intersects(o) for o in self.obstacles):
                    continue
                self.obstacles.append(rect)
                self.obstacle_rects.append((x, y, w, h))
                self._obstacle_boundaries.append(rect.boundary)
                
                speed = float(self.np_random.uniform(*self.dynamic_speed_range))
                angle = float(self.np_random.uniform(0, 2 * np.pi))
                self.obstacle_vels.append(speed * np.array([np.cos(angle), np.sin(angle)]))
                break

    # ==================================================================
    # LIDAR
    # ==================================================================
    def _cast_ray(self, angle: float) -> float:
        ox, oy = self.robot_pos
        dx, dy = np.cos(angle), np.sin(angle)
        
        # Robust inverse directions
        if abs(dx) < 1e-9:
            inv_dx = 1e9 if dx >= 0 else -1e9
        else:
            inv_dx = 1.0 / dx
            
        if abs(dy) < 1e-9:
            inv_dy = 1e9 if dy >= 0 else -1e9
        else:
            inv_dy = 1.0 / dy

        min_t = float(self.sensor_range)
        
        # Check intersection with walls
        tx1 = (0.0 - ox) * inv_dx
        tx2 = (self.world_w - ox) * inv_dx
        ty1 = (0.0 - oy) * inv_dy
        ty2 = (self.world_h - oy) * inv_dy
        
        tmin_w = max(min(tx1, tx2), min(ty1, ty2))
        tmax_w = min(max(tx1, tx2), max(ty1, ty2))
        
        if tmax_w >= tmin_w and tmin_w > 0.0:
            if tmin_w < min_t:
                min_t = tmin_w
                
        # Check intersection with obstacles
        for (x, y, w, h) in self.obstacle_rects:
            tx1 = (x - ox) * inv_dx
            tx2 = (x + w - ox) * inv_dx
            ty1 = (y - oy) * inv_dy
            ty2 = (y + h - oy) * inv_dy
            
            tmin = max(min(tx1, tx2), min(ty1, ty2))
            tmax = min(max(tx1, tx2), max(ty1, ty2))
            
            if tmax >= tmin and tmin > 0.0:
                if tmin < min_t:
                    min_t = tmin
                    
        return float(min_t)

    def _get_lidar(self) -> np.ndarray:
        dists = np.empty(self.num_sensors, dtype=np.float64)
        if self.num_sensors <= 0:
            return np.zeros(0, dtype=np.float32)

        for i in range(self.num_sensors):
            a = self.robot_angle + i * (2.0 * np.pi / self.num_sensors)
            dists[i] = self._cast_ray(a)
        self.lidar_distances = dists.copy()
        return (dists / self.sensor_range).astype(np.float32)

    # ==================================================================
    # Angle helpers
    # ==================================================================
    @staticmethod
    def _angle_diff(a1: float, a2: float) -> float:
        """Signed shortest angle from *a1* to *a2* in [-pi, pi]."""
        d = (a2 - a1) % (2.0 * np.pi)
        if d > np.pi:
            d -= 2.0 * np.pi
        return d

    # ==================================================================
    # Observation
    # ==================================================================
    def _get_obs(self) -> np.ndarray:
        lidar = self._get_lidar()                               # [0..N-1]

        # Relative goal in robot-local frame
        delta = self.goal_pos - self.robot_pos
        cos_a, sin_a = np.cos(-self.robot_angle), np.sin(-self.robot_angle)
        local_dx = delta[0] * cos_a - delta[1] * sin_a
        local_dy = delta[0] * sin_a + delta[1] * cos_a
        diag = np.sqrt(self.world_w ** 2 + self.world_h ** 2)

        # Scalar distance to goal (normalised)
        goal_dist = float(np.linalg.norm(delta)) / diag

        # Angle to goal relative to heading (normalised [-1, 1])
        goal_angle = np.arctan2(delta[1], delta[0])
        angle_diff = self._angle_diff(self.robot_angle, goal_angle) / np.pi

        # Previous action (normalised to [0, 1])
        if self.action_mode in ("continuous", "car_continuous"):
            pa = float(np.linalg.norm(self._prev_action)) / np.sqrt(2.0) if self._prev_action is not None else 0.0
        elif self.action_mode == "extended":
            pa = (self._prev_action if self._prev_action is not None else 0) / 5.0
        elif self.action_mode == "car":
            pa = (self._prev_action if self._prev_action is not None else 0) / 4.0
        else:
            pa = (self._prev_action if self._prev_action is not None else 0) / 2.0

        # Step progress (urgency signal)
        step_norm = self.step_count / self.max_steps

        # Min LIDAR (danger level)
        min_lidar = float(np.min(lidar))

        # Speed (normalised)
        speed_norm = self.robot_speed / self.step_size

        extras = np.array(
            [local_dx / diag, local_dy / diag, goal_dist, angle_diff,
             pa, step_norm, min_lidar, speed_norm],
            dtype=np.float32,
        )
        obs = np.concatenate([lidar, extras])
        return np.clip(obs, self.observation_space.low,
                       self.observation_space.high).astype(np.float32)

    # ==================================================================
    # Collision / goal checks
    # ==================================================================
    def _check_collision(self) -> bool:
        x, y = self.robot_pos
        r = self.robot_radius
        if x - r < 0 or x + r > self.world_w or y - r < 0 or y + r > self.world_h:
            return True
            
        for (ox, oy, w, h) in self.obstacle_rects:
            # Find the closest point on the AABB to the circle center
            closest_x = max(ox, min(x, ox + w))
            closest_y = max(oy, min(y, oy + h))
            dx = x - closest_x
            dy = y - closest_y
            # If distance from circle center to closest point is less than radius, collision
            if dx*dx + dy*dy <= r*r:
                return True
        return False

    def _goal_reached(self) -> bool:
        return float(np.linalg.norm(self.robot_pos - self.goal_pos)) < (
            self.robot_radius + self.goal_radius
        )

    # ==================================================================
    # Gymnasium API
    # ==================================================================
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        # Domain Randomization (±15% to physics)
        self.step_size = self.base_step_size * float(self.np_random.uniform(0.85, 1.15))
        self.turn_angle = self.base_turn_angle * float(self.np_random.uniform(0.85, 1.15))
        self.sensor_range = self.base_sensor_range * float(self.np_random.uniform(0.85, 1.15))

        margin = 60
        self.robot_pos = np.array([
            float(self.np_random.uniform(margin, self.world_w * 0.30)),
            float(self.np_random.uniform(margin, self.world_h - margin)),
        ])
        self.robot_angle = float(self.np_random.uniform(0, 2 * np.pi))

        self.goal_pos = np.array([
            float(self.np_random.uniform(self.world_w * 0.70, self.world_w - margin)),
            float(self.np_random.uniform(margin, self.world_h - margin)),
        ])

        self._generate_obstacles()
        self.step_count = 0
        self.robot_speed = 0.0
        
        self.lidar_distances = np.full(self.num_sensors, self.sensor_range)
        self.lidar_intersections = [None] * self.num_sensors

        self.step_count = 0
        self._prev_goal_dist = float(np.linalg.norm(self.robot_pos - self.goal_pos))
        self._prev_action = None
        self._smoothed_action = None
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        prev_pos = self.robot_pos.copy()
        prev_action = self._prev_action
        self._prev_action = action

        # ---- Dynamic Obstacles ----
        if self.dynamic_obstacles:
            for i in range(len(self.obstacle_rects)):
                x, y, w, h = self.obstacle_rects[i]
                vx, vy = self.obstacle_vels[i]
                x += vx
                y += vy
                # Bounce off walls
                if x < 0 or x + w > self.world_w:
                    vx = -vx
                    x = max(0.0, min(x, self.world_w - w))
                if y < 0 or y + h > self.world_h:
                    vy = -vy
                    y = max(0.0, min(y, self.world_h - h))
                self.obstacle_rects[i] = (x, y, w, h)
                self.obstacle_vels[i] = np.array([vx, vy])
                rect = shapely_box(x, y, x + w, y + h)
                self.obstacles[i] = rect
                self._obstacle_boundaries[i] = rect.boundary

        # ---- Execute action ----
        if self.action_mode == "continuous":
            lin, ang = action
            self.robot_angle += ang * self.turn_angle
            self.robot_pos = self.robot_pos + lin * self.step_size * np.array(
                [np.cos(self.robot_angle), np.sin(self.robot_angle)]
            )
        elif self.action_mode == "car_continuous":
            raw_accel, raw_steer = action
            
            # Action Smoothing (Low-Pass Filter) - made less sluggish
            if self._smoothed_action is None:
                self._smoothed_action = np.array([raw_accel, raw_steer], dtype=float)
            else:
                self._smoothed_action = 0.5 * self._smoothed_action + 0.5 * np.array([raw_accel, raw_steer])
                
            accel, steer = self._smoothed_action
            
            # Nerf reverse acceleration so driving backward is slower
            if accel < 0:
                accel *= 0.4
                
            # Update speed with acceleration and simple friction
            self.robot_speed += accel * (self.step_size * 0.5)
            self.robot_speed *= 0.90  # friction decay
            self.robot_speed = float(np.clip(self.robot_speed, -self.step_size, self.step_size))
            
            # Steer (steering effect is proportional to speed for realistic car physics, 
            # though we allow a tiny bit of turn-in-place for RL ease if needed)
            speed_ratio = abs(self.robot_speed) / self.step_size
            self.robot_angle += steer * self.turn_angle * max(0.2, speed_ratio)
            
            self.robot_pos = self.robot_pos + self.robot_speed * np.array([np.cos(self.robot_angle), np.sin(self.robot_angle)])
            
        elif self.action_mode == "car":
            if action == 0:  # Forward
                self.robot_pos = self.robot_pos + self.step_size * np.array([np.cos(self.robot_angle), np.sin(self.robot_angle)])
            elif action == 1:  # Backward
                self.robot_pos = self.robot_pos - self.step_size * np.array([np.cos(self.robot_angle), np.sin(self.robot_angle)])
            elif action == 2:  # Turn Left
                self.robot_angle -= self.turn_angle
            elif action == 3:  # Turn Right
                self.robot_angle += self.turn_angle
            elif action == 4:  # Stop
                pass  # Do nothing
        elif self.action_mode == "extended":
            if action == 0:  # FW
                self.robot_pos = self.robot_pos + self.step_size * np.array([np.cos(self.robot_angle), np.sin(self.robot_angle)])
            elif action == 1:  # L
                self.robot_angle -= self.turn_angle
            elif action == 2:  # R
                self.robot_angle += self.turn_angle
            elif action == 3:  # BW
                self.robot_pos = self.robot_pos - self.step_size * np.array([np.cos(self.robot_angle), np.sin(self.robot_angle)])
            elif action == 4:  # FW-L
                self.robot_angle -= self.turn_angle
                self.robot_pos = self.robot_pos + self.step_size * np.array([np.cos(self.robot_angle), np.sin(self.robot_angle)])
            elif action == 5:  # FW-R
                self.robot_angle += self.turn_angle
                self.robot_pos = self.robot_pos + self.step_size * np.array([np.cos(self.robot_angle), np.sin(self.robot_angle)])
        else: # basic
            if action == 0:  # Forward
                self.robot_pos = self.robot_pos + self.step_size * np.array(
                    [np.cos(self.robot_angle), np.sin(self.robot_angle)]
                )
            elif action == 1:  # Turn left (CCW on screen)
                self.robot_angle -= self.turn_angle
            elif action == 2:  # Turn right (CW on screen)
                self.robot_angle += self.turn_angle

        self.robot_angle %= 2.0 * np.pi

        # ---- Compute observation (updates LIDAR) ----
        obs = self._get_obs()

        # ---- Reward computation ----
        reward = self.rw_step                       # base step penalty
        terminated = False
        truncated = self.step_count >= self.max_steps

        if self._check_collision():
            reward = self.rw_collision
            terminated = True

        elif self._goal_reached():
            time_bonus = max(0.0, 1.0 - self.step_count / self.max_steps)
            reward = self.rw_goal + self.rw_speed * time_bonus
            terminated = True

        else:
            dist = float(np.linalg.norm(self.robot_pos - self.goal_pos))
            self._prev_goal_dist = dist

            # 1. Velocity Vector Reward
            # Calculate actual movement vector
            movement_vector = self.robot_pos - prev_pos
            movement_dist = np.linalg.norm(movement_vector)
            
            if movement_dist > 0.001:
                # Calculate vector pointing to goal
                goal_vector = self.goal_pos - prev_pos
                goal_dist = np.linalg.norm(goal_vector)
                if goal_dist > 0:
                    goal_direction = goal_vector / goal_dist
                    # Dot product measures how much of the movement was directly toward the goal
                    velocity_toward_goal = np.dot(movement_vector, goal_direction)
                    reward += self.rw_vel * velocity_toward_goal

            # 2. Clear Path Incentive
            # Reward for staying in open spaces (minimizing proximity to obstacles)
            if self.num_sensors > 0:
                min_sensor = float(np.min(self.lidar_distances)) / self.sensor_range
                reward += self.rw_clear * min_sensor

                # 3. Proximity danger penalty
                if self.rw_prox_thresh > 0.0 and min_sensor < self.rw_prox_thresh:
                    reward -= self.rw_prox * (1.0 - min_sensor / self.rw_prox_thresh)

            # 4. Spin, reverse, and jerk penalty
            if self.action_mode in ("continuous", "car_continuous"):
                # Continuous: penalize high jerk (squared change in action)
                if prev_action is not None:
                    action_np = np.array(action)
                    prev_action_np = np.array(prev_action)
                    jerk = np.sum((action_np - prev_action_np) ** 2)
                    reward -= self.rw_jerk * float(jerk)

                # Continuous: penalize high angular velocity without linear velocity
                if abs(action[1]) > 0.5 and abs(action[0]) < 0.2:
                    reward -= self.rw_spin
                # Penalize driving backwards heavily
                if getattr(self, "robot_speed", 0.0) < -0.1:
                    reward -= 0.5
            elif self.action_mode == "car":
                # Car: actions 2=Turn Left, 3=Turn Right are turns
                is_turn = action in (2, 3)
                was_turn = prev_action is not None and prev_action in (2, 3)
                if is_turn and was_turn:
                    reward -= self.rw_spin
            elif self.action_mode == "extended":
                # Extended: actions 1=Left, 2=Right are pure turns
                is_turn = action in (1, 2)
                was_turn = prev_action is not None and prev_action in (1, 2)
                if is_turn and was_turn:
                    reward -= self.rw_spin
            else:
                # Basic: actions 1=Left, 2=Right are turns
                is_turn = action in (1, 2)
                was_turn = prev_action is not None and prev_action in (1, 2)
                if is_turn and was_turn:
                    reward -= self.rw_spin
        info = {
            "is_success": self._goal_reached(),
            "is_collision": self._check_collision()
        }

        return obs, float(reward), terminated, truncated, info

    def render(self):
        """Render the environment to an rgb_array for Pygame visualization."""
        if self.render_mode != "rgb_array":
            return None

        import pygame
        # Create a surface
        surf = pygame.Surface((self.world_w, self.world_h))
        surf.fill((12, 12, 24))  # BG

        # Grid
        for gx in range(0, self.world_w, 40):
            pygame.draw.line(surf, (22, 22, 40), (gx, 0), (gx, self.world_h))
        for gy in range(0, self.world_h, 40):
            pygame.draw.line(surf, (22, 22, 40), (0, gy), (self.world_w, gy))
        pygame.draw.rect(surf, (50, 50, 80), (0, 0, self.world_w, self.world_h), 3)

        # Obstacles
        for (ox, oy, ow, oh) in self.obstacle_rects:
            pygame.draw.rect(surf, (220, 65, 55), (int(ox), int(oy), int(ow), int(oh)), border_radius=5)
            pygame.draw.rect(surf, (180, 50, 40), (int(ox), int(oy), int(ow), int(oh)), 2, border_radius=5)

        # Goal
        g = self.goal_pos.astype(int)
        pygame.draw.circle(surf, (46, 204, 113), g.tolist(), int(self.goal_radius))
        pygame.draw.circle(surf, (255, 255, 255), g.tolist(), int(self.goal_radius), 2)

        # LIDAR rays
        rx, ry = int(self.robot_pos[0]), int(self.robot_pos[1])
        ns = self.num_sensors
        if ns > 0:
            for i in range(ns):
                angle = self.robot_angle + i * (2.0 * np.pi / ns)
            dist = self.lidar_distances[i]
            ex = rx + int(dist * np.cos(angle))
            ey = ry + int(dist * np.sin(angle))
            t = max(0.0, min(1.0, dist / self.sensor_range))
            r_col = int(235 + (241 - 235) * t)
            g_col = int(77 + (196 - 77) * t)
            b_col = int(65 + (15 - 65) * t)
            pygame.draw.line(surf, (r_col, g_col, b_col), (rx, ry), (ex, ey), 1)

        # Robot
        pygame.draw.circle(surf, (41, 128, 185), (rx, ry), int(self.robot_radius) + 3)
        pygame.draw.circle(surf, (52, 152, 219), (rx, ry), int(self.robot_radius))
        hx = rx + int((self.robot_radius + 8) * np.cos(self.robot_angle))
        hy = ry + int((self.robot_radius + 8) * np.sin(self.robot_angle))
        pygame.draw.line(surf, (240, 240, 255), (rx, ry), (hx, hy), 3)
        pygame.draw.circle(surf, (240, 240, 255), (hx, hy), 3)

        return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))
