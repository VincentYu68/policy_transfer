import gym
import numpy as np
from gym import error, spaces

class OSIEnvWrapper:
    def __init__(self, env, osi, osi_hist, up_dim):
        self.wrapped_env = env
        self.env = env.env # skip a wrapper for retaining other apis
        self.osi = osi
        self.osi_hist = osi_hist
        self.up_dim = up_dim

        high = np.inf * np.ones(int(self.env.obs_dim / osi_hist + up_dim))
        low = -high
        self.observation_space = spaces.Box(low, high)

        self.action_space = env.action_space

    def process_raw_obs(self, raw_o):
        one_obs_len = int((len(raw_o) - len(self.env.control_bounds[0]) * self.env.include_act_history) / self.env.include_obs_history)
        pred_mu = self.osi.predict(raw_o)[0]
        return np.concatenate([raw_o[0:one_obs_len], pred_mu])

    def step(self, a):
        raw_o, r, d, dict = self.wrapped_env.step(a)

        return self.process_raw_obs(raw_o), r, d, dict

    def reset(self):
        raw_o = self.wrapped_env.reset()
        return self.process_raw_obs(raw_o)

    def render(self):
        return self.wrapped_env.render()

    @property
    def seed(self):
        return self.env.seed

    def pad_action(self, a):
        return self.env.pad_action(a)

    def unpad_action(self, a):
        return self.env.unpad_action(a)

    def about_to_contact(self):
        return self.env.about_to_contact()

    def state_vector(self):
        return self.env.state_vector()

    def set_state_vector(self, s):
        self.env.set_state_vector(s)

    def set_sim_parameters(self, pm):
        self.env.set_sim_parameters(pm)

    def get_sim_parameters(self):
        return self.env.get_sim_parameters()