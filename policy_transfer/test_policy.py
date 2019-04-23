__author__ = 'yuwenhao'

import matplotlib

matplotlib.use('Agg')

import gym
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import sys, os, time, errno

import joblib
import numpy as np

import matplotlib.pyplot as plt
from gym import wrappers
import tensorflow as tf
from policy_transfer.policies import mlp_policy
from policy_transfer.policies.mirror_policy import *
from policy_transfer.utils.common import *
import policy_transfer.envs

np.random.seed(11)


def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                hid_size=64, num_hid_layers=3)


def policy_mirror_fn(name, ob_space, ac_space, obs_perm, act_perm, soft_mirror):
    return MirrorPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                        hid_size=64, num_hid_layers=3, observation_permutation=obs_perm,
                        action_permutation=act_perm, soft_mirror=soft_mirror)


if __name__ == '__main__':


    append_o = []#[0.4458616 , 0.63732893, 0.98086248, 0.94058195, 0.01685923]

    if len(sys.argv) > 1:
        if sys.argv[1] == 'Minitaur':
            from pybullet_envs.minitaur.envs import minitaur_reactive_env
            from gym.wrappers import time_limit

            env = time_limit.TimeLimit(minitaur_reactive_env.MinitaurReactiveEnv(render=True,
                                                                                 accurate_motor_model_enabled=True,
                                                                                 urdf_version='rainbow_dash_v0',
                                                                                 train_UP=len(append_o) > 0,
                                                                                 resample_MP=False),
                                       max_episode_steps=1000)
        else:
            env = gym.make(sys.argv[1])
    else:
        env = gym.make('DartWalker3dRestricted-v1')

    if len(append_o) > 0 and sys.argv[1] != 'Minitaur':
        from gym import spaces

        env.env.obs_dim += len(append_o)
        high = np.inf * np.ones(env.env.obs_dim)
        low = -high
        env.env.observation_space = spaces.Box(low, high)
        env.observation_space = spaces.Box(low, high)

    if hasattr(env.env, 'disableViewer'):
        env.env.disableViewer = False
    env.env.visualize = True
    env.env.paused = False

    sess = tf.InteractiveSession()

    policy = None
    if len(sys.argv) > 2:
        policy_params = joblib.load(sys.argv[2])
        ob_space = env.observation_space
        ac_space = env.action_space

        # Modify policy representation HERE
        if 'mirror' in sys.argv[2]:
            if 'softmirror' in sys.argv[2]:
                policy = policy_mirror_fn("pi", ob_space, ac_space, obs_perm=env.env.obs_perm,
                                          act_perm=env.env.act_perm, soft_mirror=True)
            else:
                policy = policy_mirror_fn("pi", ob_space, ac_space, obs_perm=env.env.obs_perm,
                                          act_perm=env.env.act_perm, soft_mirror=False)
        else:
            policy = policy_fn("pi", ob_space, ac_space)

        U.initialize()
        assign_params(policy, policy_params)

    else:
        sys.argv.append('')

    record = False
    if len(sys.argv) > 3:
        record = int(sys.argv[3]) == 1
    if record:
        env_wrapper = wrappers.Monitor(env, 'data/videos/', force=True)
    else:
        env_wrapper = env

    print('===================')

    o = env_wrapper.reset()

    rew = 0

    traj = 10
    ct = 0
    d = False
    step = 0

    while ct < traj:
        if len(append_o) > 0:
            o = np.concatenate([o, append_o])

        if policy is not None:
            act, vpred = policy.act(step < 0, o)
        else:
            act = env.action_space.sample()

        o, r, d, env_info = env_wrapper.step(act)
        rew += r

        if env.env.visualize:
            env_wrapper.render()
        step += 1

        if d:
            print('reward: ', rew, step)
            one_traj_obs = []
            step = 0
            ct += 1
            if ct >= traj:
                break
            o = env_wrapper.reset()

    print('avg rew ', rew / traj)
