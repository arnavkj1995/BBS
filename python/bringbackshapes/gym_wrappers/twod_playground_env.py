import gym
from gym import spaces
import math
import numpy as np
import cv2

from bringbackshapes.twod_playground.arena import Arena

class TwoDPlaygroundEnv(gym.Env):
    """
    A gym enviroment to enable training on the 2D playground
    for the task of pushing all objects in the arena into the goal.
    """
    def __init__(
        self,
        action_repeat=2,
        dense_reward=False,
        shape=(64, 64),
        time_limit=None,
        render_game=False,
        debug=False,
        user_overide=False,
        random_start=True,
        max_objects=5,
        max_distractors=0,
        variable_num_objects=False,
        variable_num_distractors=False,
        variable_goal_position=False,
        agent_view_size=125,
        arena_scale=1.0
    ):
        super(TwoDPlaygroundEnv, self).__init__()
        self.env = Arena(
            debug=debug,
            user_overide=user_overide,
            render_game=render_game,
            random_start=random_start,
            max_objects=max_objects,
            max_distractors=max_distractors,
            variable_num_objects=variable_num_objects,
            variable_num_distractors=variable_num_distractors,
            variable_goal_position=variable_goal_position,
            agent_view_size=agent_view_size,
            arena_scale=arena_scale
        )
        self.debug = debug
        self.action_repeat = action_repeat
        self.dense_reward = dense_reward
        self.random_start = random_start
        self.time_limit = time_limit
        self.time_limit_step = 0
        self.shape = shape
        self.action_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([2 * math.pi, self.env.max_acc], dtype=np.float32),
        )
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.shape + (3,), dtype=np.uint8
        )

    def get_obs(self):
        observation = self.env.get_observation()
        observation = cv2.resize(observation, dsize=self.shape)
        return observation

    def step(self, action):
        self.env.check_running_status()
        self.old_dones = self.env.dones
        action_repeat_steps = 0
        done = False
        reward = 0.0
        while not done and action_repeat_steps < self.action_repeat:
            if self.env.add_brownian_distractor:
                self.env.move_distractor(self.env.num_distractors)
            self.env.apply_action(action)
            self.env.update()
            r, done = self.compute_reward()
            self.time_limit_step += 1
            action_repeat_steps += 1
            reward += r
        observation = self.get_obs()
        info = {}
        return observation, reward, done, info

    def compute_reward(self):
        # entered goal area = 1
        # reward = indicator function for num objs in goal area
        scored_objects = self.env.dones - self.old_dones
        self.old_dones = self.env.dones
        if self.dense_reward and scored_objects > 0:
            time_bonus = np.maximum(0, 1.5 - (self.env.steps / 1000))
            reward = scored_objects + time_bonus
        else:
            reward = scored_objects
        done = (
            self.env.dones == self.env.num_objects
            or not self.env.running
            or (
                self.time_limit is not None
                and self.time_limit_step > self.time_limit
            )
        )
        reward = float(reward)
        if self.debug and reward > 0:
            print("Reward components: ", scored_objects, time_bonus)
        if self.debug and done:
            print(
                "Done components: ",
                self.env.dones,
                self.env.running,
                self.time_limit_step,
            )
        return reward, done

    def reset(self):
        self.time_limit_step = 0
        self.old_dones = 0
        self.env.clear_arena()
        self.env.reset_num_objects_and_distractors()
        self.env.reset_entities()
        return self.get_obs()

    def render(self, mode="human"):
        return self.env.get_observation()

    def close(self):
        self.env.close()


class NormalizeActions(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low),
            np.isfinite(env.action_space.high),
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def action(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return original
