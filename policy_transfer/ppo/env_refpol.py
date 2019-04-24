import gym
import numpy as np
from gym import error, spaces

class EnvRefPolicy:
    def __init__(self, base_env, ref_policy, action_range=[-0.2, 0.2]):
        self.base_env = base_env
        self.ref_policy = ref_policy

        self.UP_dim = len(base_env.env.param_manager.activated_param)
        self.obs_dim = base_env.env.obs_dim - self.UP_dim

        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)
        self.action_space = base_env.action_space

        self.action_range = action_range
        self.reward_range = self.base_env.reward_range
        self.metadata = self.base_env.metadata
        self.spec = self.base_env.spec

    def seed(self, s):
        self.base_env.seed(s)

    def step(self, a):
        ref_act, _ = self.ref_policy.act(False, self.last_obs)
        act = ref_act + np.clip(a, self.action_range[0], self.action_range[1])

        o, r, d, info = self.base_env.step(act)
        self.last_obs = np.copy(o)
        return o[0:self.obs_dim], r, d, info

    def reset(self):
        self.last_obs = self.base_env.reset()
        return self.last_obs[0:self.obs_dim]