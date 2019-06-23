__author__ = 'yuwenhao'

import numpy as np
from policy_transfer.envs.dart import dart_env

##############################################################################################################
################################  Hopper #####################################################################
##############################################################################################################



class hopperContactMassManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.2, 1.0] # friction range
        self.restitution_range = [0.0, 0.3]
        self.mass_range = [2.0, 15.0]
        self.damping_range = [0.5, 3.0]
        self.power_range = [100, 300]
        self.ankle_range = [60, 300]
        self.velrew_weight_range = [-1.0, 1.0]
        self.com_offset_range = [-0.05, 0.05]
        self.frame_skip_range = [4, 10]

        self.activated_param = [0, 1, 2, 5, 9]#[0, 2,3,4,5, 6,7,8, 9, 12,13,14,15]
        self.controllable_param = [0, 1, 2, 5, 9]#[0, 2,3,4,5, 6,7,8, 9, 12,13,14,15]
        
        self.binned_param = 0 # don't bin if = 0

        self.param_dim = len(self.activated_param)
        self.sampling_selector = None
        self.selector_target = -1

    def get_simulator_parameters(self):
        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        cur_rest = self.simulator.dart_world.skeletons[0].bodynodes[0].restitution_coeff()
        restitution_param = (cur_rest - self.restitution_range[0]) / (self.restitution_range[1] - self.restitution_range[0])

        mass_param = []
        for bid in range(2, 6):
            cur_mass = self.simulator.robot_skeleton.bodynodes[bid].m
            mass_param.append((cur_mass - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0]))

        damp_param = []
        for jid in range(3, 6):
            cur_damp = self.simulator.robot_skeleton.joints[jid].damping_coefficient(0)
            damp_param.append((cur_damp - self.damping_range[0]) / (self.damping_range[1] - self.damping_range[0]))

        cur_power = self.simulator.action_scale
        power_param = (cur_power[0] - self.power_range[0]) / (self.power_range[1] - self.power_range[0])

        cur_ank_power = self.simulator.action_scale[2]
        ank_power_param = (cur_ank_power - self.ankle_range[0]) / (self.ankle_range[1] - self.ankle_range[0])

        cur_velrew_weight = self.simulator.velrew_weight
        velrew_param = (cur_velrew_weight - self.velrew_weight_range[0]) / (self.velrew_weight_range[1] - self.velrew_weight_range[0])

        com_param = []
        for bid in range(2, 6):
            if bid != 5:
                cur_com = self.simulator.robot_skeleton.bodynodes[bid].local_com()[1] - self.simulator.initial_local_coms[bid][1]
            else:
                cur_com = self.simulator.robot_skeleton.bodynodes[bid].local_com()[0] - self.simulator.initial_local_coms[bid][0]
            com_param.append((cur_com - self.com_offset_range[0]) / (self.com_offset_range[1] - self.com_offset_range[0]))

        cur_frameskip = self.simulator.frame_skip
        frameskip_param = (cur_frameskip - self.frame_skip_range[0]) / (self.frame_skip_range[1] - self.frame_skip_range[0])

        params = np.array([friction_param, restitution_param]+ mass_param + damp_param +
                          [power_param, ank_power_param, velrew_param] + com_param + [frameskip_param])[self.activated_param]
        if self.binned_param > 0:
            for i in range(len(params)):
                params[i] = int(params[i] / (1.0 / self.binned_param)) * (1.0/self.binned_param) + 0.5 / self.binned_param
        return params

    def set_simulator_parameters(self, x):
        cur_id = 0
        if 0 in self.controllable_param:
            friction = x[cur_id] * (self.range[1] - self.range[0]) + self.range[0]
            self.simulator.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(friction)
            cur_id += 1
        if 1 in self.controllable_param:
            restitution = x[cur_id] * (self.restitution_range[1] - self.restitution_range[0]) + self.restitution_range[0]
            self.simulator.dart_world.skeletons[0].bodynodes[0].set_restitution_coeff(restitution)
            self.simulator.dart_world.skeletons[1].bodynodes[-1].set_restitution_coeff(1.0)
            cur_id += 1
        for bid in range(2, 6):
            if bid in self.controllable_param:
                mass = x[cur_id] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0]
                self.simulator.robot_skeleton.bodynodes[bid].set_mass(mass)
                cur_id += 1
        for jid in range(6, 9):
            if jid in self.controllable_param:
                damp = x[cur_id] * (self.damping_range[1] - self.damping_range[0]) + self.damping_range[0]
                self.simulator.robot_skeleton.joints[jid - 3].set_damping_coefficient(0, damp)
                cur_id += 1
        if 9 in self.controllable_param:
            power = x[cur_id] * (self.power_range[1] - self.power_range[0]) + self.power_range[0]
            self.simulator.action_scale = np.array([power, power, power])
            cur_id += 1
        if 10 in self.controllable_param:
            ankpower = x[cur_id] * (self.ankle_range[1] - self.ankle_range[0]) + self.ankle_range[0]
            self.simulator.action_scale[2] = ankpower
            cur_id += 1
        if 11 in self.controllable_param:
            velrew_weight = x[cur_id] * (self.velrew_weight_range[1] - self.velrew_weight_range[0]) + self.velrew_weight_range[0]
            self.simulator.velrew_weight = velrew_weight
            cur_id += 1

        for bid in range(2, 6):
            if bid+10 in self.controllable_param:
                com = x[cur_id] * (self.com_offset_range[1] - self.com_offset_range[0]) + self.com_offset_range[0]
                init_com = np.copy(self.simulator.initial_local_coms[bid])
                if bid != 5:
                    init_com[1] += com
                else:
                    init_com[0] += com
                self.simulator.robot_skeleton.bodynodes[bid].set_local_com(init_com)
                cur_id += 1

        if 16 in self.controllable_param:
            frame_skip = x[cur_id] * (self.frame_skip_range[1] - self.frame_skip_range[0]) + self.frame_skip_range[0]
            self.simulator.frame_skip = int(frame_skip)
            cur_id += 1

    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
        if self.sampling_selector is not None:
            while not self.sampling_selector.classify(np.array([x])) == self.selector_target:
                x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        self.set_simulator_parameters(x)


class mjHopperManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.2, 1.0]  # friction range
        self.mass_range = [2.0, 20.0]
        self.damping_range = [0.15, 2.0]
        self.power_range = [150, 500]
        self.velrew_weight_range = [-1.0, 1.0]
        self.restitution_range = [0.5, 1.0]
        self.solimp_range = [0.8, 0.99]
        self.solref_range = [0.001, 0.02]
        self.armature_range = [0.05, 0.98]
        self.ankle_jnt_range = [0.5, 1.0]

        self.activated_param = [0, 1,2,3,4, 5,6,7, 8, 10, 11, 12, 13]
        self.controllable_param = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]

        self.param_dim = len(self.activated_param)
        self.sampling_selector = None
        self.selector_target = -1

    def get_simulator_parameters(self):
        cur_friction = self.simulator.model.geom_friction[-1][0]
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        mass_param = []
        for bid in range(1, 5):
            cur_mass = self.simulator.model.body_mass[bid]
            mass_param.append((cur_mass - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0]))

        damp_param = []
        for jid in range(3, 6):
            cur_damp = self.simulator.model.dof_damping[jid]
            damp_param.append((cur_damp - self.damping_range[0]) / (self.damping_range[1] - self.damping_range[0]))

        cur_power = self.simulator.model.actuator_gear[0][0]
        power_param = (cur_power - self.power_range[0]) / (self.power_range[1] - self.power_range[0])

        cur_velrew_weight = self.simulator.velrew_weight
        velrew_param = (cur_velrew_weight - self.velrew_weight_range[0]) / (
                self.velrew_weight_range[1] - self.velrew_weight_range[0])

        cur_restitution = self.simulator.model.geom_solref[-1][1]
        rest_param = (cur_restitution - self.restitution_range[0]) / (self.restitution_range[1] - self.restitution_range[0])

        cur_solimp = self.simulator.model.geom_solimp[-1][0]
        solimp_param = (cur_solimp - self.solimp_range[0]) / (self.solimp_range[1] - self.solimp_range[0])

        cur_solref = self.simulator.model.geom_solref[-1][0]
        solref_param = (cur_solref - self.solref_range[0]) / (self.solref_range[1] - self.solref_range[0])

        cur_armature = self.simulator.model.dof_armature[-1]
        armature_param = (cur_armature - self.armature_range[0]) / (self.armature_range[1] - self.armature_range[0])

        cur_jntlimit = self.simulator.model.jnt_range[-1][0]
        jntlimit_param = (cur_jntlimit - self.ankle_jnt_range[0]) / (self.ankle_jnt_range[1] - self.ankle_jnt_range[0])

        params = np.array([friction_param] + mass_param + damp_param + [power_param, velrew_param, rest_param
                                                                        ,solimp_param, solref_param, armature_param,
                                                                        jntlimit_param])[self.activated_param]
        return params

    def set_simulator_parameters(self, x):
        cur_id = 0
        if 0 in self.controllable_param:
            friction = x[cur_id] * (self.range[1] - self.range[0]) + self.range[0]
            self.simulator.model.geom_friction[-1][0] = friction
            cur_id += 1
        for bid in range(1, 5):
            if bid in self.controllable_param:
                mass = x[cur_id] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0]
                self.simulator.model.body_mass[bid] = mass
                cur_id += 1
        for jid in range(5, 8):
            if jid in self.controllable_param:
                damp = x[cur_id] * (self.damping_range[1] - self.damping_range[0]) + self.damping_range[0]
                self.simulator.model.dof_damping[jid - 2] = damp
                cur_id += 1
        if 8 in self.controllable_param:
            power = x[cur_id] * (self.power_range[1] - self.power_range[0]) + self.power_range[0]
            self.simulator.model.actuator_gear[0][0] = power
            self.simulator.model.actuator_gear[1][0] = power
            self.simulator.model.actuator_gear[2][0] = power
            cur_id += 1
        if 9 in self.controllable_param:
            velrew_weight = x[cur_id] * (self.velrew_weight_range[1] - self.velrew_weight_range[0]) + \
                            self.velrew_weight_range[0]
            self.simulator.velrew_weight = velrew_weight
            cur_id += 1
        if 10 in self.controllable_param:
            restitution = x[cur_id] * (self.restitution_range[1] - self.restitution_range[0]) + \
                            self.restitution_range[0]
            for bn in range(len(self.simulator.model.geom_solref)):
                self.simulator.model.geom_solref[bn][1] = restitution
            cur_id += 1
        if 11 in self.controllable_param:
            solimp = x[cur_id] * (self.solimp_range[1] - self.solimp_range[0]) + \
                            self.solimp_range[0]
            for bn in range(len(self.simulator.model.geom_solimp)):
                self.simulator.model.geom_solimp[bn][0] = solimp
                self.simulator.model.geom_solimp[bn][1] = solimp
            cur_id += 1
        if 12 in self.controllable_param:
            solref = x[cur_id] * (self.solref_range[1] - self.solref_range[0]) + \
                            self.solref_range[0]
            for bn in range(len(self.simulator.model.geom_solref)):
                self.simulator.model.geom_solref[bn][0] = solref
            cur_id += 1
        if 13 in self.controllable_param:
            armature = x[cur_id] * (self.armature_range[1] - self.armature_range[0]) + \
                            self.armature_range[0]
            for dof in range(3, 6):
                self.simulator.model.dof_armature[dof] = armature
            cur_id += 1
        if 14 in self.controllable_param:
            jntlimit = x[cur_id] * (self.ankle_jnt_range[1] - self.ankle_jnt_range[0]) + \
                            self.ankle_jnt_range[0]
            self.simulator.model.jnt_range[-1][0] = -jntlimit
            self.simulator.model.jnt_range[-1][1] = jntlimit
            cur_id += 1

    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
        if self.sampling_selector is not None:
            while not self.sampling_selector.classify(np.array([x])) == self.selector_target:
                x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        self.set_simulator_parameters(x)

class walker2dParamManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.mass_range = [2.0, 10.0]
        self.damping_range = [0.5, 3.0]
        self.friction_range = [0.2, 1.0] # friction range
        self.restitution_range = [0.0, 0.8]
        self.power_range = [50, 150]
        self.ankle_power_range = [10, 50]
        self.frame_skip_range = [4, 10]
        self.up_noise_range = [0.0, 1.0]

        self.activated_param = [7,8,9,10,11,12, 13,14]#[0,1,2,3,4,5,6,  7,8,9,10,11,12,  13, 14, 15, 16]
        self.controllable_param = [7,8,9,10,11,12, 13,14]#[0,1,2,3,4,5,6,  7,8,9,10,11,12,  13, 14, 15, 16]

        self.param_dim = len(self.activated_param)
        self.sampling_selector = None
        self.selector_target = -1

    def get_simulator_parameters(self):
        mass_param = []
        for bid in range(2, 9):
            cur_mass = self.simulator.robot_skeleton.bodynodes[bid].m
            mass_param.append((cur_mass - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0]))

        damp_param = []
        for jid in range(3, 9):
            cur_damp = self.simulator.robot_skeleton.joints[jid].damping_coefficient(0)
            damp_param.append((cur_damp - self.damping_range[0]) / (self.damping_range[1] - self.damping_range[0]))\

        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.friction_range[0]) / (self.friction_range[1] - self.friction_range[0])

        cur_rest = self.simulator.dart_world.skeletons[0].bodynodes[0].restitution_coeff()
        restitution_param = (cur_rest - self.restitution_range[0]) / (self.restitution_range[1] - self.restitution_range[0])

        cur_power = self.simulator.action_scale[0]
        power_param = (cur_power - self.power_range[0]) / (self.power_range[1] - self.power_range[0])

        cur_ankl_power = self.simulator.action_scale[2]
        ank_power_param = (cur_ankl_power - self.ankle_power_range[0]) / (self.ankle_power_range[1] - self.ankle_power_range[0])

        cur_frameskip = self.simulator.frame_skip
        frameskip_param = (cur_frameskip - self.frame_skip_range[0]) / (
                self.frame_skip_range[1] - self.frame_skip_range[0])

        cur_up_noise = self.simulator.UP_noise_level
        up_noise_param = (cur_up_noise - self.up_noise_range[0]) / (self.up_noise_range[1] - self.up_noise_range[0])

        return np.array(mass_param+damp_param+[friction_param, restitution_param, power_param, ank_power_param,
                                               frameskip_param,  up_noise_param])[self.activated_param]

    def set_simulator_parameters(self, x):
        cur_id = 0

        for bid in range(0, 7):
            if bid in self.controllable_param:
                mass = x[cur_id] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0]
                self.simulator.robot_skeleton.bodynodes[bid+2].set_mass(mass)
                cur_id += 1
        for jid in range(7, 13):
            if jid in self.controllable_param:
                damp = x[cur_id] * (self.damping_range[1] - self.damping_range[0]) + self.damping_range[0]
                self.simulator.robot_skeleton.joints[jid - 4].set_damping_coefficient(0, damp)
                cur_id += 1

        if 13 in self.controllable_param:
            friction = x[cur_id] * (self.friction_range[1] - self.friction_range[0]) + self.friction_range[0]
            self.simulator.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(friction)
            cur_id += 1
        if 14 in self.controllable_param:
            restitution = x[cur_id] * (self.restitution_range[1] - self.restitution_range[0]) + self.restitution_range[0]
            self.simulator.dart_world.skeletons[0].bodynodes[0].set_restitution_coeff(restitution)
            self.simulator.dart_world.skeletons[1].bodynodes[-1].set_restitution_coeff(1.0)
            cur_id += 1
        if 15 in self.controllable_param:
            power = x[cur_id] * (self.power_range[1] - self.power_range[0]) + self.power_range[0]
            self.simulator.action_scale[[0,1,3,4]] = power
            cur_id += 1
        if 16 in self.controllable_param:
            ank_power = x[cur_id] * (self.ankle_power_range[1] - self.ankle_power_range[0]) + self.ankle_power_range[0]
            self.simulator.action_scale[[2,5]] = ank_power
            cur_id += 1
        if 17 in self.controllable_param:
            frame_skip = x[cur_id] * (self.frame_skip_range[1] - self.frame_skip_range[0]) + self.frame_skip_range[0]
            self.simulator.frame_skip = int(frame_skip)
            cur_id += 1
        if 18 in self.controllable_param:
            up_noise = x[cur_id] * (self.up_noise_range[1] - self.up_noise_range[0]) + self.up_noise_range[0]
            self.simulator.UP_noise_level = up_noise
            cur_id += 1


    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
        self.set_simulator_parameters(x)

class mjWalkerParamManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.mass_range = [2.0, 15.0]
        self.range = [0.5, 2.0]  # friction range
        self.restitution_range = [0.5, 1.0]
        self.solimp_range = [0.8, 0.99]
        self.solref_range = [0.001, 0.02]
        self.armature_range = [0.05, 0.98]
        self.tilt_z_range = [-0.18, 0.18]

        self.activated_param = [0]
        self.controllable_param = [0]

        self.param_dim = len(self.activated_param)
        self.sampling_selector = None
        self.selector_target = -1

    def get_simulator_parameters(self):
        mass_param = []
        for bid in range(1, 8):
            cur_mass = self.simulator.model.body_mass[bid]
            mass_param.append((cur_mass - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0]))

        cur_friction = self.simulator.model.geom_friction[-1][0]
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        cur_restitution = self.simulator.model.geom_solref[-1][1]
        rest_param = (cur_restitution - self.restitution_range[0]) / (self.restitution_range[1] - self.restitution_range[0])

        cur_solimp = self.simulator.model.geom_solimp[-1][0]
        solimp_param = (cur_solimp - self.solimp_range[0]) / (self.solimp_range[1] - self.solimp_range[0])

        cur_solref = self.simulator.model.geom_solref[-1][0]
        solref_param = (cur_solref - self.solref_range[0]) / (self.solref_range[1] - self.solref_range[0])

        cur_armature = self.simulator.model.dof_armature[-1]
        armature_param = (cur_armature - self.armature_range[0]) / (self.armature_range[1] - self.armature_range[0])

        cur_tiltz = self.simulator.tilt_z
        tiltz_param = (cur_tiltz - self.tilt_z_range[0]) / (self.tilt_z_range[1] - self.tilt_z_range[0])

        params = np.array(mass_param + [friction_param, rest_param ,solimp_param, solref_param, armature_param, tiltz_param])[self.activated_param]
        return params

    def set_simulator_parameters(self, x):
        cur_id = 0
        for bid in range(0, 7):
            if bid in self.controllable_param:
                mass = x[cur_id] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0]
                self.simulator.model.body_mass[bid] = mass
                cur_id += 1

        if 7 in self.controllable_param:
            friction = x[cur_id] * (self.range[1] - self.range[0]) + self.range[0]
            self.simulator.model.geom_friction[-1][0] = friction
            cur_id += 1
        if 8 in self.controllable_param:
            restitution = x[cur_id] * (self.restitution_range[1] - self.restitution_range[0]) + \
                            self.restitution_range[0]
            for bn in range(len(self.simulator.model.geom_solref)):
                self.simulator.model.geom_solref[bn][1] = restitution
            cur_id += 1
        if 9 in self.controllable_param:
            solimp = x[cur_id] * (self.solimp_range[1] - self.solimp_range[0]) + \
                            self.solimp_range[0]
            for bn in range(len(self.simulator.model.geom_solimp)):
                self.simulator.model.geom_solimp[bn][0] = solimp
                self.simulator.model.geom_solimp[bn][1] = solimp
            cur_id += 1
        if 10 in self.controllable_param:
            solref = x[cur_id] * (self.solref_range[1] - self.solref_range[0]) + \
                            self.solref_range[0]
            for bn in range(len(self.simulator.model.geom_solref)):
                self.simulator.model.geom_solref[bn][0] = solref
            cur_id += 1
        if 11 in self.controllable_param:
            armature = x[cur_id] * (self.armature_range[1] - self.armature_range[0]) + \
                            self.armature_range[0]
            for dof in range(3, 6):
                self.simulator.model.dof_armature[dof] = armature
            cur_id += 1
        if 12 in self.controllable_param:
            tiltz = x[cur_id] * (self.tilt_z_range[1] - self.tilt_z_range[0]) + self.tilt_z_range[0]
            self.simulator.tilt_z = tiltz
            self.simulator.model.opt.gravity[:] = [9.81 * np.sin(tiltz), 0.0, -9.81 * np.cos(tiltz)]
            cur_id += 1


    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
        if self.sampling_selector is not None:
            while not self.sampling_selector.classify(np.array([x])) == self.selector_target:
                x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        self.set_simulator_parameters(x)

class cheetahParamManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.mass_range = [1.0, 15.0]
        self.damping_range = [1.5, 10.0]
        self.stiff_range = [50, 300]
        self.friction_range = [0.2, 1.0] # friction range
        self.restitution_range = [0.0, 0.5]
        self.gact_scale_range = [0.3, 1.5]
        self.tilt_z_range = [-0.78, 0.78]

        self.activated_param = [0,1,2,3,4,5,6,7]
        self.controllable_param = [0,1,2,3,4,5,6,7]

        self.param_dim = len(self.activated_param)
        self.sampling_selector = None
        self.selector_target = -1

    def get_simulator_parameters(self):
        mass_param = []
        for bid in range(2, 10):
            cur_mass = self.simulator.robot_skeleton.bodynodes[bid].m
            mass_param.append((cur_mass - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0]))

        damp_param = []
        for jid in [4,5,6,7,8,9]:
            cur_damp = self.simulator.robot_skeleton.joints[jid].damping_coefficient(0)
            damp_param.append((cur_damp - self.damping_range[0]) / (self.damping_range[1] - self.damping_range[0]))

        stiff_param = []
        for jid in [4, 5, 6, 7, 8, 9]:
            cur_stiff = self.simulator.robot_skeleton.joints[jid].spring_stiffness(0)
            stiff_param.append((cur_stiff - self.stiff_range[0]) / (self.stiff_range[1] - self.stiff_range[0]))

        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.friction_range[0]) / (self.friction_range[1] - self.friction_range[0])

        cur_rest = self.simulator.dart_world.skeletons[0].bodynodes[0].restitution_coeff()
        restitution_param = (cur_rest - self.restitution_range[0]) / (self.restitution_range[1] - self.restitution_range[0])

        cur_power = self.simulator.g_action_scaler
        power_param = (cur_power - self.gact_scale_range[0]) / (self.gact_scale_range[1] - self.gact_scale_range[0])

        cur_tiltz = self.simulator.tilt_z
        tiltz_param = (cur_tiltz - self.tilt_z_range[0]) / (self.tilt_z_range[1] - self.tilt_z_range[0])


        return np.array(mass_param+damp_param+stiff_param+[friction_param, restitution_param, power_param, tiltz_param])[self.activated_param]

    def set_simulator_parameters(self, x):
        cur_id = 0

        for bid in range(0, 8):
            if bid in self.controllable_param:
                mass = x[cur_id] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0]
                self.simulator.robot_skeleton.bodynodes[bid+2].set_mass(mass)
                cur_id += 1
        for id, jid in enumerate([4,5,6,7,8,9]):
            if id+8 in self.controllable_param:
                damp = x[cur_id] * (self.damping_range[1] - self.damping_range[0]) + self.damping_range[0]
                self.simulator.robot_skeleton.joints[jid].set_damping_coefficient(0, damp)
                cur_id += 1

        for id, jid in enumerate([4,5,6,7,8,9]):
            if id+14 in self.controllable_param:
                stiff = x[cur_id] * (self.stiff_range[1] - self.stiff_range[0]) + self.stiff_range[0]
                self.simulator.robot_skeleton.joints[jid].set_spring_stiffness(0, stiff)
                cur_id += 1

        if 20 in self.controllable_param:
            friction = x[cur_id] * (self.friction_range[1] - self.friction_range[0]) + self.friction_range[0]
            self.simulator.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(friction)
            cur_id += 1
        if 21 in self.controllable_param:
            restitution = x[cur_id] * (self.restitution_range[1] - self.restitution_range[0]) + self.restitution_range[0]
            self.simulator.dart_world.skeletons[0].bodynodes[0].set_restitution_coeff(restitution)
            self.simulator.dart_world.skeletons[1].bodynodes[-1].set_restitution_coeff(1.0)
            cur_id += 1
        if 22 in self.controllable_param:
            power = x[cur_id] * (self.gact_scale_range[1] - self.gact_scale_range[0]) + self.gact_scale_range[0]
            self.simulator.g_action_scaler = power
            cur_id += 1

        if 23 in self.controllable_param:
            tiltz = x[cur_id] * (self.tilt_z_range[1] - self.tilt_z_range[0]) + self.tilt_z_range[0]
            self.simulator.tilt_z = tiltz
            self.simulator.dart_world.set_gravity([9.81 * np.sin(tiltz), -9.81 * np.cos(tiltz), 0.0])
            cur_id += 1

    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
        self.set_simulator_parameters(x)


class mjcheetahParamManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.2, 1.0]  # friction range
        self.velrew_weight_range = [-1.0, 1.0]
        self.restitution_range = [0.5, 1.0]
        self.solimp_range = [0.8, 0.99]
        self.solref_range = [0.001, 0.02]
        self.armature_range = [0.05, 0.98]
        self.tilt_z_range = [-0.18, 0.18]

        self.activated_param = [5]
        self.controllable_param = [5]

        self.param_dim = len(self.activated_param)
        self.sampling_selector = None
        self.selector_target = -1

    def get_simulator_parameters(self):
        cur_friction = self.simulator.model.geom_friction[-1][0]
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        cur_velrew_weight = self.simulator.velrew_weight
        velrew_param = (cur_velrew_weight - self.velrew_weight_range[0]) / (
                self.velrew_weight_range[1] - self.velrew_weight_range[0])

        cur_restitution = self.simulator.model.geom_solref[-1][1]
        rest_param = (cur_restitution - self.restitution_range[0]) / (self.restitution_range[1] - self.restitution_range[0])

        cur_solimp = self.simulator.model.geom_solimp[-1][0]
        solimp_param = (cur_solimp - self.solimp_range[0]) / (self.solimp_range[1] - self.solimp_range[0])

        cur_solref = self.simulator.model.geom_solref[-1][0]
        solref_param = (cur_solref - self.solref_range[0]) / (self.solref_range[1] - self.solref_range[0])

        cur_armature = self.simulator.model.dof_armature[-1]
        armature_param = (cur_armature - self.armature_range[0]) / (self.armature_range[1] - self.armature_range[0])

        cur_tiltz = self.simulator.tilt_z
        tiltz_param = (cur_tiltz - self.tilt_z_range[0]) / (self.tilt_z_range[1] - self.tilt_z_range[0])

        params = np.array([friction_param, velrew_param, rest_param ,solimp_param, solref_param, armature_param, tiltz_param])[self.activated_param]
        return params

    def set_simulator_parameters(self, x):
        cur_id = 0
        if 0 in self.controllable_param:
            friction = x[cur_id] * (self.range[1] - self.range[0]) + self.range[0]
            self.simulator.model.geom_friction[-1][0] = friction
            cur_id += 1
        if 1 in self.controllable_param:
            velrew_weight = x[cur_id] * (self.velrew_weight_range[1] - self.velrew_weight_range[0]) + \
                            self.velrew_weight_range[0]
            self.simulator.velrew_weight = velrew_weight
            cur_id += 1
        if 2 in self.controllable_param:
            restitution = x[cur_id] * (self.restitution_range[1] - self.restitution_range[0]) + \
                            self.restitution_range[0]
            for bn in range(len(self.simulator.model.geom_solref)):
                self.simulator.model.geom_solref[bn][1] = restitution
            cur_id += 1
        if 3 in self.controllable_param:
            solimp = x[cur_id] * (self.solimp_range[1] - self.solimp_range[0]) + \
                            self.solimp_range[0]
            for bn in range(len(self.simulator.model.geom_solimp)):
                self.simulator.model.geom_solimp[bn][0] = solimp
                self.simulator.model.geom_solimp[bn][1] = solimp
            cur_id += 1
        if 4 in self.controllable_param:
            solref = x[cur_id] * (self.solref_range[1] - self.solref_range[0]) + \
                            self.solref_range[0]
            for bn in range(len(self.simulator.model.geom_solref)):
                self.simulator.model.geom_solref[bn][0] = solref
            cur_id += 1
        if 5 in self.controllable_param:
            armature = x[cur_id] * (self.armature_range[1] - self.armature_range[0]) + \
                            self.armature_range[0]
            for dof in range(3, 6):
                self.simulator.model.dof_armature[dof] = armature
            cur_id += 1
        if 6 in self.controllable_param:
            tiltz = x[cur_id] * (self.tilt_z_range[1] - self.tilt_z_range[0]) + self.tilt_z_range[0]
            self.simulator.tilt_z = tiltz
            self.simulator.model.opt.gravity[:] = [9.81 * np.sin(tiltz), 0.0, -9.81 * np.cos(tiltz)]
            cur_id += 1


    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
        if self.sampling_selector is not None:
            while not self.sampling_selector.classify(np.array([x])) == self.selector_target:
                x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        self.set_simulator_parameters(x)
