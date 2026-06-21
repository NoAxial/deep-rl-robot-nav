"""
curriculum.py -- Performance-Based Curriculum Callback.

Gradually scales the difficulty of the environment based on the agent's 
success rate over the last N episodes, rather than a fixed timestep schedule.
"""

from __future__ import annotations

import collections
import logging
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class PerformanceCurriculumCallback(BaseCallback):
    def __init__(self, raw_envs, window_size: int = 50, advance_threshold: float = 0.8, demote_threshold: float = 0.4, verbose: int = 1):
        super().__init__(verbose)
        if isinstance(raw_envs, list):
            self.raw_envs = raw_envs
        else:
            self.raw_envs = [raw_envs]
            
        self.window_size = window_size
        self.advance_threshold = advance_threshold
        self.demote_threshold = demote_threshold
        
        # Track history of success (True/False)
        self.history = collections.deque(maxlen=window_size)
        
        self.current_stage = 1
        self.max_stage = 10
        self.cooldown = 0
        
    def _apply_stage(self):
        # Linearly interpolate difficulty based on stage 1 to 10
        t = (self.current_stage - 1) / max(1, self.max_stage - 1)
        
        # Obstacles: from 1-2 to 8-12
        obs_min = int(1 + t * 7)
        obs_max = int(2 + t * 10)
        
        # Goal radius: from 30 to 10
        goal_radius = 30.0 - t * 20.0
        
        # Dynamic obstacles turn on at stage 5 (halfway)
        dynamic = self.current_stage >= 5
        
        for env in self.raw_envs:
            env.obs_min_n = obs_min
            env.obs_max_n = obs_max
            env.goal_radius = goal_radius
            env.dynamic_obstacles = dynamic
        
        if self.verbose > 0:
            success_rate = np.mean(self.history) if len(self.history) > 0 else 0.0
            print(f"\n[Curriculum] >>> MOVED TO STAGE {self.current_stage}/{self.max_stage} (Success Rate: {success_rate*100:.1f}%) <<<")
            print(f"             Obstacles: {obs_min}-{obs_max}")
            print(f"             Goal Radius: {goal_radius:.1f}")
            print(f"             Dynamic Obstacles: {dynamic}")
            
    def _on_training_start(self) -> None:
        self._apply_stage()
        
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        
        for idx, done in enumerate(dones):
            if done:
                # Add success to history
                is_success = infos[idx].get("is_success", False)
                self.history.append(is_success)
                
                # Check for stage advancement or demotion if history is full
                if len(self.history) == self.window_size and self.cooldown <= 0:
                    success_rate = np.mean(self.history)
                    
                    if success_rate >= self.advance_threshold and self.current_stage < self.max_stage:
                        self.current_stage += 1
                        self.cooldown = self.window_size // 2  # wait before changing again
                        self.history.clear()
                        self._apply_stage()
                    elif success_rate <= self.demote_threshold and self.current_stage > 1:
                        self.current_stage -= 1
                        self.cooldown = self.window_size // 2
                        self.history.clear()
                        self._apply_stage()
                else:
                    self.cooldown -= 1
                    
        return True
