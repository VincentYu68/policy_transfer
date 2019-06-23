import numpy as np
from gym import utils
from policy_transfer.envs.dart import dart_env
from policy_transfer.envs.dart.parameter_managers import *


class DartWalker2dEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0]*6,[-1.0]*6])
        self.action_scale = np.array([100, 100, 20, 100, 100, 20])
        obs_dim = 17
        self.train_UP = False
        self.noisy_input = True
        self.resample_MP = False
        self.UP_noise_level = 0.0
        self.param_manager = walker2dParamManager(self)

        self.use_sparse_reward = False

        self.include_obs_history = 1
        self.include_act_history = 0
        obs_dim *= self.include_obs_history
        obs_dim += len(self.control_bounds[0]) * self.include_act_history

        obs_perm_base = np.array(
            [0.0001, 1, 5, 6, 7, 2, 3, 4, 8, 9, 10, 14, 15, 16, 11, 12, 13])
        act_perm_base = np.array([3, 4, 5, 0.0001, 1, 2])
        self.obs_perm = np.copy(obs_perm_base)

        for i in range(self.include_obs_history - 1):
            self.obs_perm = np.concatenate(
                [self.obs_perm, np.sign(obs_perm_base) * (np.abs(obs_perm_base) + len(self.obs_perm))])
        for i in range(self.include_act_history):
            self.obs_perm = np.concatenate(
                [self.obs_perm, np.sign(act_perm_base) * (np.abs(act_perm_base) + len(self.obs_perm))])
        self.act_perm = np.array([3, 4, 5, 0.0001, 1, 2])

        if self.train_UP:
            obs_dim += len(self.param_manager.activated_param)
            self.obs_perm = np.concatenate([self.obs_perm, np.arange(int(len(self.obs_perm)),
                                                int(len(self.obs_perm)+len(self.param_manager.activated_param)))])

        dart_env.DartEnv.__init__(self, ['walker2d.skel'], 4, obs_dim, self.control_bounds, disableViewer=True)

        self.dart_worlds[0].set_collision_detector(3)

        self.dart_world=self.dart_worlds[0]
        self.robot_skeleton=self.dart_world.skeletons[-1]
        if not self.disableViewer:
            self._get_viewer().sim = self.dart_world

        self.cur_step = 0

        # data structure for modeling delays in observation and action
        self.observation_buffer = []
        self.action_buffer = []
        self.state_buffer = []

        self.obs_delay = 0
        self.act_delay = 0

        self.current_param = self.param_manager.get_simulator_parameters()

        utils.EzPickle.__init__(self)

    def about_to_contact(self):
        return False

    def pad_action(self, a):
        full_ac = np.zeros(len(self.robot_skeleton.q))
        full_ac[3:] = a
        return full_ac

    def pre_advance(self):
        self.posbefore = self.robot_skeleton.q[0]

    def terminated(self):
        s = self.state_vector()
        height = self.robot_skeleton.bodynodes[2].com()[1]
        ang = self.robot_skeleton.q[2]
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                   (height > .8) and (height < 2.0) and (abs(ang) < 1.0))
        if self.cur_step >= 1000:
            done = True
        return done

    def post_advance(self):
        pass

    def reward_func(self, a, step_skip=1, sparse=False):
        posafter, ang = self.robot_skeleton.q[0, 2]
        height = self.robot_skeleton.bodynodes[2].com()[1]

        alive_bonus = 1.0 * step_skip
        vel = (posafter - self.posbefore) / self.dt
        reward = vel
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        joint_limit_penalty = 0
        for j in [-2, -5]:
            if (self.robot_skeleton.q_lower[j] - self.robot_skeleton.q[j]) > -0.05:
                joint_limit_penalty += abs(1.5)
            if (self.robot_skeleton.q_upper[j] - self.robot_skeleton.q[j]) < 0.05:
                joint_limit_penalty += abs(1.5)

        reward -= 5e-1 * joint_limit_penalty

        if sparse:
            reward = 0.0
            if self.terminated():
                reward = self.robot_skeleton.q[0]

        return reward

    def advance(self, a):
        self.action_buffer.append(np.copy(a))
        if len(self.action_buffer) < self.act_delay + 1:
            a *= 0
        else:
            a = self.action_buffer[-self.act_delay - 1]

        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[3:] = clamped_control * self.action_scale
        self.posbefore = self.robot_skeleton.q[0]

        # compensate for gravity
        #tau[1] = self.robot_skeleton.mass() * 9.81

        self.do_simulation(tau, self.frame_skip)

    def step(self, a):
        self.cur_step += 1
        self.advance(a)
        reward = self.reward_func(a, sparse=self.use_sparse_reward)

        done = self.terminated()

        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            self.robot_skeleton.dq
        ])
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]

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
                final_obs = np.concatenate([final_obs, [0.0] * len(self.control_bounds[0])])

        if self.train_UP:
            UP_parameters = self.param_manager.get_simulator_parameters()
            final_obs = np.concatenate([final_obs, UP_parameters])
        if self.noisy_input:
            final_obs = final_obs + np.random.normal(0, .01, len(final_obs))

        return final_obs

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.00015, high=.00015, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.00015, high=.00015, size=self.robot_skeleton.ndofs)
        qpos[3] += 0.5
        self.set_state(qpos, qvel)

        if self.resample_MP:
            self.param_manager.resample_parameters()
            self.current_param = self.param_manager.get_simulator_parameters()

        self.observation_buffer = []
        self.action_buffer = []
        self.cur_step = 0

        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -4
