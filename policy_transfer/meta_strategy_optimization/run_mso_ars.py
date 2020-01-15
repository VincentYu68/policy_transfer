#!/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
import policy_transfer.envs
from baselines import logger
import sys
import joblib, pickle
import tensorflow as tf
import numpy as np
from mpi4py import MPI
from policy_transfer.policies.mirror_policy import *
from policy_transfer.policies.mlp_policy import MlpPolicy
from policy_transfer.utils.common import *


config_name=''
output_interval = 100


def callback(localv, globalv):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(config_name)
    save_dict = {}
    variables = localv['pi'].get_variables()
    for i in range(len(variables)):
        cur_val = variables[i].eval()
        save_dict[variables[i].name] = cur_val
    pickle.dump(save_dict, open(logger.get_dir() + '/policy_params_pickle' + '.pkl', 'wb'))

    if localv['iters_so_far'] % output_interval != 0:
        return
    pickle.dump(save_dict, open(logger.get_dir()+'/policy_params_pickle_'+str(localv['iters_so_far'])+'.pkl', 'wb'))



def train(env_id, max_iter, inner_iter, seed, skilldim, tasknum, warmstart, mirror, dyn_params):
    from policy_transfer.meta_strategy_optimization import ars_mso
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed+MPI.COMM_WORLD.Get_rank())

    env = gym.make(env_id)

    env.env.param_manager.activated_param = dyn_params
    env.env.param_manager.controllable_param = dyn_params

    if hasattr(env.env, 'obs_perm') and skilldim > 0:
        cur_perm = env.env.obs_perm
        beginid = len(cur_perm)
        obs_perm_base = np.concatenate([cur_perm, np.arange(beginid, beginid + skilldim)])
        env.env.obs_perm = obs_perm_base

    with open(logger.get_dir()+"/envinfo.txt", "w") as text_file:
        text_file.write(str(env.env.__dict__))

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)

    def policy_mirror_fn(name, ob_space, ac_space):
        return MirrorPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2, observation_permutation=env.env.env.obs_perm,
                            action_permutation=env.env.env.act_perm, soft_mirror=False)

    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), "monitor.json"), allow_early_resets=True)
    with open(logger.get_dir()+"/config.txt", "w") as text_file:
        text_file.write(str(locals()))

    if hasattr(env.env.env, "param_manager"):
        with open(logger.get_dir()+"/params.txt", "w") as text_file:
            text_file.write(str(env.env.env.param_manager.__dict__))

    env.seed(seed+MPI.COMM_WORLD.Get_rank())

    gym.logger.setLevel(logging.WARN)

    pol_func = policy_fn
    if mirror:
        pol_func = policy_mirror_fn

    if len(warmstart) != 0:
        if 'pickle' in warmstart:
            warstart_params = pickle.load(open(warmstart, 'rb'))
        else:
            warstart_params = joblib.load(warmstart)
    else:
        warstart_params = None

    ars_mso.ars_optimize(env, pol_func,
                 perturb_mag=0.02,
                 learning_rate=0.005,
                 eval_epoch=1,
                 params_per_thread=8,
                 top_perturb=8,
                 maxiter=max_iter,
                 callback=callback,
                 init_policy_params=warstart_params,
                 skilldim=skilldim,
                 task_num=tasknum,
                 inner_iters=inner_iter,
    )

    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='DartHopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--name', help='name of experiments', type=str, default="")
    parser.add_argument('--max_iter', help='maximum iteration number', type=int, default = 1000)
    parser.add_argument('--inner_iter', help='inner iteration number', type=int, default=30)
    parser.add_argument('--output_interval', help='interval of outputting policies', type=int, default=100)
    parser.add_argument('--warmstart', help='warmstart of experiments', type=str, default="")
    parser.add_argument('--skilldim', help='dimension of latent variable', type=int, default=2)
    parser.add_argument('--task_number', help='number of tasks to sample per iteration', type=int, default=5)
    parser.add_argument('--mirror', help='use mirror policy', default="False")
    parser.add_argument('--dyn_params', action='append', type=int)


    args = parser.parse_args()
    global config_name, output_interval
    output_interval = args.output_interval
    logger.reset()
    config_name = 'data/mso_ars_'+args.env+str(args.seed)+'_'+args.name

    config_name += '_skilldim' + str(args.skilldim)
    config_name += '_maxiter' + str(args.max_iter)
    config_name += '_tasknum' + str(args.task_number)
    config_name += '_inneriter' + str(args.inner_iter)
    if len(args.warmstart) > 0:
        config_name += '_warmstart'

    if args.mirror == 'True':
        config_name += '_mirror'


    logger.configure(config_name, ['json','stdout'])
    train(args.env, skilldim=args.skilldim, max_iter=int(args.max_iter), inner_iter=int(args.inner_iter),
          seed=args.seed, tasknum = int(args.task_number), warmstart=args.warmstart, mirror=args.mirror=='True',
          dyn_params = args.dyn_params)

if __name__ == '__main__':
    main()
