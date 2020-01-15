from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from baselines.common.cg import cg
from policy_transfer.policy_transfer_strategy_optimization.up_optimizer import UPOptimizer

import os, errno
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from policy_transfer.utils.common import *
import copy

###################################################
### Evaluate the performance of one sample params
###################################################
def evaluate_performance(env, pol, params, set_pol_from_flat, eval_epoch, task, optimized_embedding):
    performances = []
    set_pol_from_flat(params)
    obs = []
    env.env.env.set_task(task)
    for i in range(eval_epoch):
        data,_,reward = run_one_traj(env, pol, False, observation_app=optimized_embedding)
        performances.append(reward)
        obs += data['obs']
    return obs, np.mean(performances), eval_epoch



def ars_optimize(env, policy_func, perturb_mag, learning_rate, eval_epoch,
                 params_per_thread, top_perturb, maxiter, policy_scope="pi",
                 init_policy_params = None, eval_func = evaluate_performance, callback=None,
                 skilldim=3, task_num=5, inner_iters=10
                 ):
    ob_space = env.observation_space
    ac_space = env.action_space
    from gym import spaces
    obs_dim_base = ob_space.low.shape[0] + skilldim
    high = np.inf * np.ones(obs_dim_base)
    low = -high
    skill_ob_space = spaces.Box(low, high)

    np.random.seed(MPI.COMM_WORLD.Get_rank() * 1023+1)

    pi = policy_func(policy_scope, skill_ob_space, ac_space)  # Construct network for new policy

    pol_var_list = [v for v in pi.get_trainable_variables() if 'pol' in v.name and 'logstd' not in v.name]
    pol_var_size = np.sum([np.prod(v.shape) for v in pol_var_list])
    print(pol_var_list)
    print('pol_var_size: ', pol_var_size)

    get_pol_flat = U.GetFlat(pol_var_list)
    set_pol_from_flat = U.SetFromFlat(pol_var_list)

    adam = MpiAdam(pol_var_list)

    U.initialize()

    adam.sync()

    if init_policy_params is not None:
        cur_scope = pi.get_variables()[0].name[0:pi.get_variables()[0].name.find('/')]
        orig_scope = list(init_policy_params.keys())[0][0:list(init_policy_params.keys())[0].find('/')]
        print(cur_scope, orig_scope)
        for i in range(len(pi.get_variables())):
            if pi.get_variables()[i].name.replace(cur_scope, orig_scope, 1) in init_policy_params:
                assign_op = pi.get_variables()[i].assign(
                    init_policy_params[pi.get_variables()[i].name.replace(cur_scope, orig_scope, 1)])
                tf.get_default_session().run(assign_op)

    # Used to construct the task-embedding mapping
    skill_optimizer = UPOptimizer(env, pi, skilldim, eval_num=1, verbose=False, bayesian_opt=True)

    param_dim = len(get_pol_flat())
    episodes_so_far = 0
    iters_so_far = 0
    current_parameters = np.copy(get_pol_flat())
    for it in range(maxiter):
        logger.log("********** Iteration %i ************" % it)

        if callback is not None:
            callback(locals(), globals())
        current_parameters = np.copy(get_pol_flat())

        perturbations = []
        performances = []
        all_obs = []

        if it % inner_iters == 0:
            logger.log("==== Reconstruct task set at iteration %i ======" % it)
            # Construct the task_embedding first
            task_embeddings = []
            for task_id in range(int(np.max([task_num / MPI.COMM_WORLD.Get_size(), 1]))):
                task_parameters = env.env.env.resample_task()
                optimized_embedding = None
                skill_optimizer.reset()
                if skilldim > 0:
                    skill_optimizer.optimize(maxiter=20, max_steps=50000, custom_bound=[-1.0, 1.0])
                    optimized_embedding = skill_optimizer.best_x
                    print(task_parameters, optimized_embedding, skill_optimizer.best_f)
                else:
                    optimized_embedding = []
                task_embeddings.append([task_parameters, optimized_embedding])
                print(optimized_embedding)
            all_task_embeddings = MPI.COMM_WORLD.allgather(task_embeddings)
            task_embeddings = []
            for te in all_task_embeddings:
                task_embeddings += te
            task_embeddings = task_embeddings[0:task_num]

        for sp in range(params_per_thread):
            sampled_perturbation = np.random.normal(0, 1, param_dim)

            positive_perf = []
            negative_perf = []
            for task_emb in task_embeddings:
                pert_param = current_parameters + sampled_perturbation * perturb_mag
                obs, positive_pert_performance, episodes = eval_func(env, pi, pert_param, set_pol_from_flat, eval_epoch, task_emb[0], task_emb[1])
                all_obs += obs
                episodes_so_far += episodes
                positive_perf.append(positive_pert_performance)

                pert_param = current_parameters - sampled_perturbation * perturb_mag
                obs, negative_pert_performance, episodes = eval_func(env, pi, pert_param, set_pol_from_flat, eval_epoch, task_emb[0], task_emb[1])
                all_obs += obs
                episodes_so_far += episodes
                negative_perf.append(negative_pert_performance)

            perturbations.append(sampled_perturbation)
            performances.append([np.mean(positive_perf), np.mean(negative_perf)])
        all_performances = np.concatenate(MPI.COMM_WORLD.allgather(performances), axis=0)
        all_perturbations = np.concatenate(MPI.COMM_WORLD.allgather(perturbations), axis=0)

        max_perf_list = np.max(all_performances, axis=1)
        max_perf_list.sort()
        top_k_perf = max_perf_list[-top_perturb]

        parameter_update = current_parameters * 0
        utilized_rewards = []
        for sp in range(len(all_performances)):
            if np.max(all_performances[sp]) >= top_k_perf:
                parameter_update += all_perturbations[sp] * (all_performances[sp][0] - all_performances[sp][1])
                utilized_rewards.append(all_performances[sp][0])
                utilized_rewards.append(all_performances[sp][1])
        parameter_update /= top_perturb

        lr_scaler = np.std(utilized_rewards)

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(np.array(all_obs))  # update running mean/std for policy

        delta_parameters = parameter_update * learning_rate / lr_scaler
        current_parameters += delta_parameters
        set_pol_from_flat(current_parameters)
        total_episodes_sofar = np.sum(MPI.COMM_WORLD.allgather(episodes_so_far))
        final_performance = []
        for task_emb in task_embeddings:
            obs, perf, episodes = eval_func(env, pi, current_parameters, set_pol_from_flat, eval_epoch, task_emb[0], task_emb[1])
            final_performance.append(perf)
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.record_tabular('EpRewMean', np.mean(final_performance))
            logger.record_tabular('EpisodesSoFar', total_episodes_sofar)
            logger.record_tabular('Parameter magnitude', np.linalg.norm(current_parameters))
            logger.record_tabular('Delta magnitude', np.linalg.norm(delta_parameters))
            logger.dump_tabular()
        iters_so_far += 1

    print('Optimized parameter: ', current_parameters)


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
