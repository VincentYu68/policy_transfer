import numpy as np
from gym import utils
from policy_transfer.envs.mujoco import mujoco_env
from policy_transfer.envs.dart.parameter_managers import *
from gym import error, spaces

class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.train_UP = False
        self.noisy_input = True

        self.resample_MP = False  # whether to resample the model paraeters
        self.param_manager = mjHopperManager(self)
        self.velrew_weight = 1.0

        self.include_obs_history = 1
        self.include_act_history = 0

        # data structure for modeling delays in observation and action
        self.observation_buffer = []
        self.action_buffer = []
        self.obs_delay = 0
        self.act_delay = 0

        self.cur_step = 0

        self.use_sparse_reward = False
        self.horizon = 999

        self.total_reward = 0

        self.lowlevel_policy = None
        self.lowlevel_policy_dim = 5
        self.highlevel_action_space = spaces.Box(low=np.array([-1] * self.lowlevel_policy_dim),
                                                 high=np.array([1] * self.lowlevel_policy_dim))

        self.add_perturbation = False
        self.perturbation_parameters = [0.2, 300, 2, 20]  # probability, magnitude, bodyid, duration

        mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 4)


        utils.EzPickle.__init__(self)

    def pad_action(self, a):
        full_ac = np.zeros(len(self.sim.data.qpos))
        full_ac[3:] = a
        return full_ac

    def unpad_action(self, a):
        return a[3:]

    def advance(self, a):
        self.action_buffer.append(np.copy(a))
        if len(self.action_buffer) < self.act_delay + 1:
            a *= 0
        else:
            a = self.action_buffer[-self.act_delay - 1]

        self.do_simulation(a, self.frame_skip)

    def about_to_contact(self):
        return False

    def post_advance(self):
        pass

    def terminated(self):
        s = self.state_vector()
        height, ang = self.sim.data.qpos[1:3]
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .8))
        if self.cur_step >= self.horizon:
            done = True
        return done

    def pre_advance(self):
        self.posbefore = self.sim.data.qpos[0]

    def reward_func(self, a):
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - self.posbefore) / self.dt * self.velrew_weight
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        joint_limit_penalty = 0
        if np.abs(self.sim.data.qpos[-2]) < 0.05:
            joint_limit_penalty += 1.5
        #reward -= 5e-1 * joint_limit_penalty
        if self.use_sparse_reward:
            self.total_reward += reward
            reward = 0.0
            if self.terminated():
                reward = self.total_reward

        return reward

    def step(self, a):
        if self.lowlevel_policy is not None:
            # a = 0.2*a + np.array([0.09471898, 0.2101066 , 0.99302079, 0.51291866, 0.01000209])
            a, _ = self.lowlevel_policy.act(False, np.concatenate([self._get_obs(), a]))
        self.cur_step += 1
        self.pre_advance()
        self.advance(a)
        self.post_advance()

        reward = self.reward_func(a)

        done = self.terminated()

        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self, update_buffer = True):
        state = np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])
        if self.train_UP:
            state = np.concatenate([state, self.param_manager.get_simulator_parameters()])
        if self.noisy_input:
            state = state + np.random.normal(0, .01, len(state))

        if update_buffer:
            self.observation_buffer.append(np.copy(state))

        final_obs = np.array([])
        for i in range(self.include_obs_history):
            if self.obs_delay + i < len(self.observation_buffer):
                final_obs = np.concatenate([final_obs, self.observation_buffer[-self.obs_delay - 1 - i]])
            else:
                final_obs = np.concatenate([final_obs, self.observation_buffer[0] * 0.0])

        for i in range(self.include_act_history):
            if i < len(self.action_buffer):
                final_obs = np.concatenate([final_obs, self.action_buffer[-1 - i]])
            else:
                final_obs = np.concatenate([final_obs, [0.0] * 3])

        return final_obs

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.01, high=.01, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.01, high=.01, size=self.model.nv)

        self.set_state(qpos, qvel)

        self.observation_buffer = []
        self.action_buffer = []

        self.cur_step = 0
        self.total_reward = 0

        if self.resample_MP:
            self.param_manager.resample_parameters()

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
