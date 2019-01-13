from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
import numpy as np

from policy_transfer.policies.utils import *
from policy_transfer.policies.utils import *

class MirrorPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True, obs_name='ob',\
              obrms=True, final_std=0.01, init_logstd=0.0, observation_permutation=None,action_permutation=None, soft_mirror=False):
        assert isinstance(ob_space, gym.spaces.Box)

        obs_perm_mat = np.zeros((len(observation_permutation), len(observation_permutation)), dtype=np.float32)
        self.obs_perm_mat=obs_perm_mat
        for i, perm in enumerate(observation_permutation):
            obs_perm_mat[i][int(np.abs(perm))] = np.sign(perm)

        if isinstance(ac_space, gym.spaces.Box):
            act_perm_mat = np.zeros((len(action_permutation), len(action_permutation)), dtype=np.float32)
            self.act_perm_mat=act_perm_mat
            for i, perm in enumerate(action_permutation):
                self.act_perm_mat[i][int(np.abs(perm))] = np.sign(perm)
        elif isinstance(ac_space, gym.spaces.MultiDiscrete):
            total_dim = int(np.sum(ac_space.nvec))
            dim_index = np.concatenate([[0], np.cumsum(ac_space.nvec)])
            act_perm_mat = np.zeros((total_dim, total_dim), dtype=np.float32)
            self.act_perm_mat = act_perm_mat
            for i, perm in enumerate(action_permutation):
                perm_mat = np.identity(ac_space.nvec[i])
                if np.sign(perm) < 0:
                    perm_mat = np.flipud(perm_mat)
                    self.act_perm_mat[dim_index[i]:dim_index[i] + ac_space.nvec[i],
                    dim_index[int(np.abs(perm))]:dim_index[int(np.abs(perm))]+ac_space.nvec[int(np.abs(perm))]] = perm_mat


        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None
        print(self.pdtype)
        print([sequence_length] + list(ob_space.shape))
        ob = U.get_placeholder(name=obs_name, dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)
        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        mirror_ob = tf.matmul(ob, obs_perm_mat)
        mirror_obz = tf.clip_by_value((mirror_ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        if not obrms:
            obz = ob
        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(dense(last_out, hid_size, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        self.vpred = dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]

        if isinstance(ac_space, gym.spaces.Box):
            pol_net = GenericFF('pol_net', ob_space.shape[0], [], pdtype.param_shape()[0]//2, hid_size, num_hid_layers)
        elif isinstance(ac_space, gym.spaces.MultiDiscrete):
            pol_net = GenericFF('pol_net', ob_space.shape[0], [], pdtype.param_shape()[0], hid_size,
                                num_hid_layers)

        orig_out = pol_net.get_output_tensor(obz, None, tf.nn.tanh)
        mirr_out = tf.matmul(pol_net.get_output_tensor(mirror_obz, None, tf.nn.tanh), act_perm_mat)
        if not soft_mirror:
            mean = orig_out + mirr_out
        else:
            mean = orig_out
            self.additional_loss = tf.reduce_mean(tf.abs(orig_out - mirr_out)) * 1.0

        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.constant_initializer(init_logstd))
            pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = mean

        self.pd = pdtype.pdfromflat(pdparam)
        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())

        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob, is_training=False):
        ac1, vpred1 = self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

