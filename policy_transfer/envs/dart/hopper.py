import numpy as np
from gym import utils
from policy_transfer.envs.dart import dart_env

from policy_transfer.envs.dart.parameter_managers import *
import copy
from gym import error, spaces

class DartHopperEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0, 1.0, 1.0],[-1.0, -1.0, -1.0]])
        self.action_scale = np.array([200.0, 200.0, 200.0]) * 1.0
        self.train_UP = True
        self.noisy_input = True

        obs_dim = 11

        self.velrew_weight = 1.0
        self.UP_noise_level = 0.0
        self.resample_MP = True  # whether to resample the model paraeters
        self.fixed_UP_obs = None

        self.param_manager = hopperContactMassManager(self)

        if self.train_UP:
            obs_dim += len(self.param_manager.activated_param)

        self.t = 0

        self.total_dist = []

        self.include_obs_history = 1
        self.include_act_history = 0
        obs_dim *= self.include_obs_history
        obs_dim += len(self.control_bounds[0]) * self.include_act_history

        dart_env.DartEnv.__init__(self, ['hopper_capsule.skel'], 4, obs_dim, self.control_bounds, disableViewer=True)

        self.initial_local_coms = [np.copy(bn.local_com()) for bn in self.robot_skeleton.bodynodes]

        self.current_param = self.param_manager.get_simulator_parameters()

        self.dart_worlds[0].set_collision_detector(3)

        self.dart_world=self.dart_worlds[0]
        self.robot_skeleton=self.dart_world.skeletons[-1]

        # data structure for modeling delays in observation and action
        self.observation_buffer = []
        self.action_buffer = []
        self.obs_delay = 0
        self.act_delay = 0

        self.param_manager.set_simulator_parameters(self.current_param)

        print('sim parameters: ', self.param_manager.get_simulator_parameters())
        self.current_param = self.param_manager.get_simulator_parameters()
        self.active_param = self.param_manager.activated_param

        # data structure for actuation modeling
        self.zeroed_height = self.robot_skeleton.bodynodes[2].com()[1]

        self.add_perturbation=False
        self.perturbation_parameters = [0.1, 800, 5, 2] # probability, magnitude, bodyid, duration

        self.learnable_perturbation = True
        self.learnable_perturbation_list = [['h_foot', 20, 20]]  # [bodynode name, force magnitude, torque magnitude
        self.learnable_perturbation_space = spaces.Box(np.array([-1] * len(self.learnable_perturbation_list) * 6),
                                                       np.array([1] * len(self.learnable_perturbation_list) * 6))
        self.learnable_perturbation_act = np.zeros(len(self.learnable_perturbation_list) * 6)

        utils.EzPickle.__init__(self)

    def pad_action(self, a):
        full_ac = np.zeros(len(self.robot_skeleton.q))
        full_ac[3:] = a
        return full_ac

    def unpad_action(self, a):
        return a[3:]

    def do_simulation(self, tau, n_frames):
        for _ in range(n_frames):
            if self.learnable_perturbation: # if learn to perturb
                for bid, pert_param in enumerate(self.learnable_perturbation_list):
                    force_dir = self.learnable_perturbation_act[bid * 6: bid * 6 + 3]
                    torque_dir = self.learnable_perturbation_act[bid * 6 + 3: bid * 6 + 6]
                    if np.all(force_dir == 0):
                        pert_force = np.zeros(3)
                    else:
                        pert_force = pert_param[1] * force_dir / np.linalg.norm(force_dir)
                    if np.all(torque_dir == 0):
                        pert_torque = np.zeros(3)
                    else:
                        pert_torque = pert_param[2] * torque_dir / np.linalg.norm(torque_dir)
                    self.robot_skeleton.bodynode(pert_param[0]).add_ext_force(pert_force)
                    self.robot_skeleton.bodynode(pert_param[0]).add_ext_torque(pert_torque)


            self.robot_skeleton.set_forces(tau)
            self.dart_world.step()

    def advance(self, a):
        self.action_buffer.append(np.copy(a))
        if len(self.action_buffer) < self.act_delay + 1:
            a *= 0
        else:
            a = self.action_buffer[-self.act_delay-1]

        self.posbefore = self.robot_skeleton.q[0]
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]

        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[3:] = clamped_control * self.action_scale
        self.do_simulation(tau, self.frame_skip)


    def about_to_contact(self):
        return False

    def post_advance(self):
        self.dart_world.check_collision()

    def terminated(self):
        self.fall_on_ground = False
        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        permitted_contact_bodies = [self.robot_skeleton.bodynodes[-1], self.robot_skeleton.bodynodes[-2]]
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()
            if contact.bodynode1 not in permitted_contact_bodies and contact.bodynode2 not in permitted_contact_bodies:
                self.fall_on_ground = True

        s = self.state_vector()
        height = self.robot_skeleton.bodynodes[2].com()[1]
        ang = self.robot_skeleton.q[2]
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (np.abs(self.robot_skeleton.dq) < 100).all()\
            #and not self.fall_on_ground)
            and (height > self.height_threshold_low) and (abs(ang) < .4))
        return done

    def pre_advance(self):
        self.posbefore = self.robot_skeleton.q[0]

    def reward_func(self, a, step_skip=1):
        posafter = self.robot_skeleton.q[0]
        alive_bonus = 1.0
        joint_limit_penalty = 0
        for j in [-2]:
            if (self.robot_skeleton.q_lower[j] - self.robot_skeleton.q[j]) > -0.05:
                joint_limit_penalty += abs(1.5)
            if (self.robot_skeleton.q_upper[j] - self.robot_skeleton.q[j]) < 0.05:
                joint_limit_penalty += abs(1.5)
        reward = (posafter - self.posbefore) / self.dt * self.velrew_weight
        reward += alive_bonus * step_skip
        reward -= 1e-3 * np.square(a).sum()
        reward -= 5e-1 * joint_limit_penalty
        return reward

    def step(self, a):
        self.t += self.dt
        self.pre_advance()
        self.advance(a)
        reward = self.reward_func(a)

        done = self.terminated()

        ob = self._get_obs()

        self.cur_step += 1

        envinfo = {}

        return ob, reward, done, envinfo

    def _get_obs(self, update_buffer = True):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            self.robot_skeleton.dq,
        ])
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]

        if self.train_UP:
            if self.fixed_UP_obs is None:
                UP = self.param_manager.get_simulator_parameters()
                if self.UP_noise_level > 0:
                    UP += np.random.uniform(-self.UP_noise_level, self.UP_noise_level, len(UP))
                    UP = np.clip(UP, -0.05, 1.05)
            else:
                UP = self.fixed_UP_obs
            state = np.concatenate([state, UP])

        if self.noisy_input:
            state = state + np.random.normal(0, .01, len(state))

        if update_buffer:
            self.observation_buffer.append(np.copy(state))

        final_obs = np.array([])
        for i in range(self.include_obs_history):
            if self.obs_delay + i < len(self.observation_buffer):
                final_obs = np.concatenate([final_obs, self.observation_buffer[-self.obs_delay-1-i]])
            else:
                final_obs = np.concatenate([final_obs, self.observation_buffer[0]*0.0])
        for i in range(self.include_act_history):
            if i < len(self.action_buffer):
                final_obs = np.concatenate([final_obs, self.action_buffer[-1-i]])
            else:
                final_obs = np.concatenate([final_obs, [0.0]*len(self.control_bounds[0])])


        return final_obs

    def reset_model(self):
        for world in self.dart_worlds:
            world.reset()
        self.zeroed_height = self.robot_skeleton.bodynodes[2].com()[1]
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)


        self.set_state(qpos, qvel)
        if self.resample_MP:
            self.param_manager.resample_parameters()
            self.current_param = self.param_manager.get_simulator_parameters()

        self.observation_buffer = []
        self.action_buffer = []

        state = self._get_obs(update_buffer = True)

        self.cur_step = 0

        self.height_threshold_low = 0.56*self.robot_skeleton.bodynodes[2].com()[1]
        self.t = 0

        self.fall_on_ground = False

        self.learnable_perturbation_act = np.zeros(len(self.learnable_perturbation_list) * 6)

        return state

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -4

    def state_vector(self):
        s = np.copy(np.concatenate([self.robot_skeleton.q, self.robot_skeleton.dq]))
        s[1] += self.zeroed_height
        return s

    def set_state_vector(self, s):
        snew = np.copy(s)
        snew[1] -= self.zeroed_height
        self.robot_skeleton.q = snew[0:len(self.robot_skeleton.q)]
        self.robot_skeleton.dq = snew[len(self.robot_skeleton.q):]

    def set_sim_parameters(self, pm):
        self.param_manager.set_simulator_parameters(pm)

    def get_sim_parameters(self):
        return self.param_manager.get_simulator_parameters()
