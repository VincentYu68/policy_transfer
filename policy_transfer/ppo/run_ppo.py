#!/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
import policy_transfer.envs
from baselines import logger
import sys
import joblib
import tensorflow as tf
import numpy as np
from mpi4py import MPI
from policy_transfer.policies.mirror_policy import *
from policy_transfer.policies.mlp_policy import MlpPolicy

output_interval = 10

def callback(localv, globalv):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(globalv.keys())
    save_dict = {}
    variables = localv['pi'].get_variables()
    for i in range(len(variables)):
        cur_val = variables[i].eval()
        save_dict[variables[i].name] = cur_val
    joblib.dump(save_dict, logger.get_dir() + '/policy_params' + '.pkl', compress=True)
    if localv['iters_so_far'] % output_interval != 0:
        return
    joblib.dump(save_dict, logger.get_dir()+'/policy_params_'+str(localv['iters_so_far'])+'.pkl', compress=True)



def train(env_id, num_timesteps, seed, batch_size, clip, schedule, mirror, warmstart, train_up, dyn_params):
    from policy_transfer.ppo import ppo_sgd
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)

    if env_id == 'Minitaur':
        from pybullet_envs.minitaur.envs import minitaur_reactive_env
        from gym.wrappers import time_limit
        env = time_limit.TimeLimit(minitaur_reactive_env.MinitaurReactiveEnv(render=False,
                                                         accurate_motor_model_enabled=True,
                                                         urdf_version='rainbow_dash_v0',
                                                         train_UP=False,
                                                         resample_MP=False), max_episode_steps=1000)
    else:
        env = gym.make(env_id)
        if train_up:
            if env.env.train_UP is not True:
                env.env.train_UP = True
                env.env.resample_MP = True
                from gym import spaces

                env.env.param_manager.activated_param = dyn_params
                env.env.param_manager.controllable_param = dyn_params
                env.env.obs_dim += len(env.env.param_manager.activated_param)

                high = np.inf * np.ones(env.env.obs_dim)
                low = -high
                env.env.observation_space = spaces.Box(low, high)
                env.observation_space = spaces.Box(low, high)

                if hasattr(env.env, 'obs_perm'):
                    obpermapp = np.arange(len(env.env.obs_perm), len(env.env.obs_perm) + len(env.env.param_manager.activated_param))
                    env.env.obs_perm = np.concatenate([env.env.obs_perm, obpermapp])

    with open(logger.get_dir()+"/envinfo.txt", "w") as text_file:
        text_file.write(str(env.env.__dict__))

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=3)

    def policy_mirror_fn(name, ob_space, ac_space):
        return MirrorPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=3, observation_permutation=env.env.env.obs_perm,
                            action_permutation=env.env.env.act_perm, soft_mirror=(mirror==2))


    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), "monitor.json"), allow_early_resets=True)
    env.seed(seed+MPI.COMM_WORLD.Get_rank())

    gym.logger.setLevel(logging.WARN)

    if mirror:
        pol_func = policy_mirror_fn
    else:
        pol_func = policy_fn

    if len(warmstart) > 0:
        warstart_params = joblib.load(warmstart)
    else:
        warstart_params = None
    ppo_sgd.learn(env, pol_func,
            max_timesteps=num_timesteps,
            timesteps_per_batch=int(batch_size),
            clip_param=clip, entcoeff=0.0,
            optim_epochs=5, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule=schedule,
                        callback=callback,
                  init_policy_params=warstart_params,
    )


    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='DartHopperPT-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--name', help='name of experiments', type=str, default="")
    parser.add_argument('--max_step', help='maximum step size', type=int, default = 1000000)
    parser.add_argument('--batch_size', help='batch size', type=int, default=4000)
    parser.add_argument('--clip', help='clip', type=float, default=0.2)
    parser.add_argument('--schedule', help='schedule', default='constant')
    parser.add_argument('--train_up', help='whether train up', default='True')
    parser.add_argument('--dyn_params', action='append', type=int)
    parser.add_argument('--output_interval', help='interval of outputting policies', type=int, default=10)
    parser.add_argument('--mirror', help='whether to use mirror, (0: not mirror, 1: hard mirror, 2: soft mirror)', type=int, default=0)
    parser.add_argument('--warmstart', help='path to warmstart policies',
                        type=str, default="")


    args = parser.parse_args()
    global output_interval
    output_interval = args.output_interval
    logger.reset()
    config_name = 'data/ppo_'+args.env+str(args.seed)+'_'+args.name

    if args.mirror == 1:
        config_name += '_mirror'
    elif args.mirror == 2:
        config_name += '_softmirror'

    if len(args.warmstart) > 0:
        config_name += '_warmstart'

    if args.train_up == 'True':
        config_name += '_UP'

    logger.configure(config_name, ['json','stdout'])
    train(args.env, num_timesteps=int(args.max_step), seed=args.seed, batch_size=args.batch_size,
          clip=args.clip, schedule=args.schedule,
          mirror=args.mirror, warmstart=args.warmstart, train_up=args.train_up=='True', dyn_params = args.dyn_params
          )

if __name__ == '__main__':
    main()
