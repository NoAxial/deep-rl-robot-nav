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
    def __init__(self, window_size: int = 50, advance_threshold: float = 0.8, demote_threshold: float = 0.3, min_episodes_per_stage: int = 100, verbose: int = 1):
        super().__init__(verbose)
            
        self.window_size = window_size
        self.advance_threshold = advance_threshold
        self.demote_threshold = demote_threshold
        self.min_episodes_per_stage = min_episodes_per_stage
        
        # Track history of success (True/False)
        self.history = collections.deque(maxlen=window_size)
        
        self.current_stage = 1
        self.max_stage = 10
        self.cooldown = 0
        self.stage_episode_count = 0
        # Grace period: freeze demotions for N episodes after any stage change
        self.grace_steps_remaining = 0
        self.grace_period = min_episodes_per_stage * 2
        
    def _apply_stage(self):
        # Linearly interpolate difficulty based on stage 1 to 10
        t = (self.current_stage - 1) / max(1, self.max_stage - 1)
        
        # Obstacles: from 1-2 to 8-12
        obs_min = int(1 + t * 7)
        obs_max = int(2 + t * 10)
        
        # Goal radius: from 30 to 10
        goal_radius = 30.0 - t * 20.0
        
        if self.training_env:
            self.training_env.set_attr('obs_min_n', obs_min)
            self.training_env.set_attr('obs_max_n', obs_max)
            self.training_env.set_attr('goal_radius', goal_radius)
        
        if self.verbose > 0:
            success_rate = np.mean(self.history) if len(self.history) > 0 else 0.0
            print(f"\n[Curriculum] >>> MOVED TO STAGE {self.current_stage}/{self.max_stage} (Success Rate: {success_rate*100:.1f}%) <<<")
            print(f"             Obstacles: {obs_min}-{obs_max}")
            print(f"             Goal Radius: {goal_radius:.1f}")
            
    def _on_training_start(self) -> None:
        self._apply_stage()
        
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        
        for idx, done in enumerate(dones):
            if done:
                self.stage_episode_count += 1
                # Add success to history
                is_success = infos[idx].get("is_success", False)
                self.history.append(is_success)
                
                # Check for stage advancement or demotion if history is full
                if len(self.history) == self.window_size and self.cooldown <= 0:
                    if self.stage_episode_count >= self.min_episodes_per_stage:
                        success_rate = np.mean(self.history)
                        
                        new_stage = self.current_stage
                        
                        # Advance if performing well
                        if success_rate >= self.advance_threshold and self.current_stage < self.max_stage:
                            new_stage += 1
                        # Only demote if not in grace period after a recent stage change
                        elif (success_rate <= self.demote_threshold
                              and self.current_stage > 1
                              and self.grace_steps_remaining <= 0):
                            new_stage -= 1

                        if new_stage != self.current_stage:
                            self.current_stage = new_stage
                            self.cooldown = self.window_size // 2
                            self.stage_episode_count = 0
                            self.history.clear()
                            self.grace_steps_remaining = self.grace_period
                            self._apply_stage()

                # Decrement cooldown and grace period once per episode
                if self.cooldown > 0:
                    self.cooldown -= 1
                if self.grace_steps_remaining > 0:
                    self.grace_steps_remaining -= 1
        return True
