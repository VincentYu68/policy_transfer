# this is to test the fine-tuned parameters in a unified way

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gym, joblib, tensorflow as tf
from baselines.ppo1 import mlp_policy
from baselines.common import tf_util as U
from gym import spaces
from baselines import logger

from policy_transfer.policy_transfer_strategy_optimization.up_optimizer import UPOptimizer
from policy_transfer.utils.common import *
from policy_transfer.policies.mirror_policy import *
import policy_transfer.envs

import numpy as np

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='DartHopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--name', help='name of experiments', type=str, default="")
    parser.add_argument('--UP_dim', help='up dimension', type=int, default=0)
    parser.add_argument('--policy_path', help='path to policy', type=str, default="")
    parser.add_argument('--run_cma', help='if False, load existing data and play policy', type=str, default="True")
    parser.add_argument('--max_step', help='maximum step allowed', type=int, default=50000)
    parser.add_argument('--sparse_rew', type=str, default='True')
    args = parser.parse_args()

    eval_repeat = 5

    # extract information from args
    name = args.name
    seed = args.seed
    UP_dim = args.UP_dim
    policy_path = args.policy_path
    run_cma = args.run_cma == 'True'
    max_step = args.max_step

    use_sparse_rew = args.sparse_rew == 'True'

    if args.env == 'Minitaur':
        from pybullet_envs.minitaur.envs import minitaur_reactive_env
        from gym.wrappers import time_limit

        obs_in = 1
        act_in = 0
        if testing_mode == 'HIST':
            obs_in = 10
            act_in = 10
        env = time_limit.TimeLimit(minitaur_reactive_env.MinitaurReactiveEnv(render=False,
                    accurate_motor_model_enabled = True,
                    urdf_version='rainbow_dash_v0',
                    include_obs_history=obs_in, include_act_history=act_in, train_UP=False, resample_MP=False), max_episode_steps=1000)
    else:
        env = gym.make(args.env)
        if hasattr(env.env, 'disableViewer'):
            env.env.disableViewer = False

    def policy_fn(name, ob_space, ac_space):
        hid_size = 64
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=hid_size, num_hid_layers=3)

    def policy_mirror_fn(name, ob_space, ac_space):
        obpermapp = np.arange(len(env.env.obs_perm), len(env.env.obs_perm)+UP_dim)
        ob_perm = np.concatenate([env.env.obs_perm, obpermapp])

        return MirrorPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                            hid_size=64, num_hid_layers=3, observation_permutation=ob_perm,
                            action_permutation=env.env.act_perm, soft_mirror=False)

    config_name = 'data/processed_evaluation_single/eval_single_' + name + '_' + args.env
    if use_sparse_rew:
        config_name += '_sparse_rew'

    logger.configure(config_name, ['json', 'stdout'])

    test_param_dict = locals()
    with open(logger.get_dir() + "/testinfo.txt", "w") as text_file:
        text_file.write(str(test_param_dict))

    high = np.inf * np.ones(len(env.observation_space.low)+UP_dim)
    low = -high
    ob_space = spaces.Box(low, high)
    ac_space = env.action_space

    #env.env.param_manager.set_simulator_parameters([0.2, 0.4, 0.5, 0.3, 0.1])

    if 'mirror' in policy_path:
        pol_fn = policy_mirror_fn
    else:
        pol_fn = policy_fn
    policy = pol_fn("pi_test", ob_space, ac_space)
    policy_params = joblib.load(policy_path)

    sess = tf.InteractiveSession()
    U.initialize()

    cur_scope = policy.get_variables()[0].name[0:policy.get_variables()[0].name.find('/')]
    orig_scope = list(policy_params.keys())[0][0:list(policy_params.keys())[0].find('/')]

    vars = policy.get_variables()
    for i in range(len(policy.get_variables())):
        assign_op = policy.get_variables()[i].assign(
            policy_params[policy.get_variables()[i].name.replace(cur_scope, orig_scope, 1)])
        sess.run(assign_op)

    optimizer = UPOptimizer(env, policy, UP_dim, eval_num=3, verbose=True)

    if use_sparse_rew:
        env.env.use_sparse_reward = True

    optimizer.reset()
    if run_cma:
        optimizer.optimize(500, max_step)
        # extract the solution list
        sol_list = optimizer.solution_history
        sample_num_list = optimizer.sample_num_history
        np.savetxt(logger.get_dir()+'/solution_list.txt', sol_list)
        np.savetxt(logger.get_dir() + '/sample_num.txt', sample_num_list)

        if hasattr(env.env, 'max_step'):
            env.env.max_step = 1000

        # use dense reward for evaluation
        if use_sparse_rew:
            env.env.use_sparse_reward = False

        reward_list = []

        for solution in sol_list:
            xopt = solution

            eval_rew = []
            for _ in range(eval_repeat):
                traj, rews, rew = run_one_traj(env, policy, stochastic=False, observation_app=xopt)
                eval_rew.append(rew)
            print('Evaluated rew: ', np.mean(eval_rew))
            reward_list.append(np.mean(eval_rew))

        plt.figure()
        plt.plot(sample_num_list, reward_list)
        plt.savefig(logger.get_dir() + '/performance_curve.jpg')

        np.savetxt(logger.get_dir() + '/eval_results.txt', reward_list)
    else:
        sol_list = np.loadtxt(logger.get_dir() + '/solution_list.pkl')
        sample_num_list = np.loadtxt(logger.get_dir() + '/sample_num.txt')

    # visualize the final policy
    run_one_traj(env, policy, stochastic=False, observation_app=sol_list[-1], render=True)















