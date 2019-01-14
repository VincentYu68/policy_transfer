from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque

import os, errno
from utils.common import *
import copy
import gc

def memory():
    import os
    import psutil
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
    logger.record_tabular("memory_use", memoryUse)
    print('memory use: ', memoryUse)


def traj_segment_generator(pi, env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    avg_vels = np.zeros(horizon, 'float32')
    acs = np.array([ac for _ in range(horizon)])


    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens, "avg_vels":avg_vels}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac

        ob, rew, new, envinfo = env.step(ac)
        rews[i] = rew
        if "avg_vel" in envinfo:
            avg_vels[i] = envinfo["avg_vel"]

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            broke = False
            if 'broke_sim' in envinfo:
                if envinfo['broke_sim']:
                    broke = True
            if not broke:
                ep_rets.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
            else:
                t = 0
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]

    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - new[t + 1]
        delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def calc_vtarg_and_adv(new, rew, vpred, nextvpred, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(new, 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred_app = np.append(vpred, nextvpred)
    T = len(rew)
    adv = gaelam = np.empty(T, 'float32')
    rew = rew
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred_app[t+1] * nonterminal - vpred_app[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    tdlamret = adv + vpred
    return adv, tdlamret

def compute_adapt_metric(seg, compute_ratios, compute_kls, adapt_threshold, metric):
    reopt = False
    if metric == 0:
        ratio_aft = compute_ratios(seg["ob"], seg["ac"])
        if np.max(ratio_aft) > adapt_threshold:  # if ratio too large
            reopt = True
    elif metric == 1:
        kl_aft = compute_kls(seg["ob"], seg["ac"])
        if np.mean(kl_aft) > adapt_threshold:
            reopt = True
    elif metric == 2:
        kl_aft = compute_kls(seg["ob"], seg["ac"])
        for dim in range(len(seg["ob"])):
            if kl_aft[dim] > adapt_threshold:  # if ratio too large
                reopt = True

    return reopt

def learn(env, policy_func, *,
        timesteps_per_batch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
          init_policy_params = None,
          policy_scope='pi'
        ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space

    pi = policy_func(policy_scope, ob_space, ac_space) # Construct network for new policy
    oldpi = policy_func("old"+policy_scope, ob_space, ac_space) # Network for old policy

    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return
    clip_tf = tf.placeholder(dtype=tf.float32)

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)

    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_tf, 1.0 + clip_tf * lrmult) * atarg
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)

    if hasattr(pi, 'additional_loss'):
        pol_surr += pi.additional_loss

    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))


    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    pol_var_list = [v for v in pi.get_trainable_variables() if
                        'placehold' not in v.name and 'offset' not in v.name and
                        'secondary' not in v.name and 'vf' not in v.name and 'pol' in v.name]
    pol_var_size = np.sum([np.prod(v.shape) for v in pol_var_list])

    get_pol_flat = U.GetFlat(pol_var_list)
    set_pol_from_flat = U.SetFromFlat(pol_var_list)

    total_loss = pol_surr + pol_entpen + vf_loss


    lossandgrad = U.function([ob, ac, atarg, ret, lrmult, clip_tf], losses + [U.flatgrad(total_loss, var_list)])
    pol_lossandgrad = U.function([ob, ac, atarg, ret, lrmult, clip_tf],
                             losses + [U.flatgrad(total_loss, pol_var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult, clip_tf], losses)
    compute_losses_cpo = U.function([ob, ac, atarg, ret, lrmult, clip_tf], [tf.reduce_mean(surr1), pol_entpen, vf_loss, meankl, meanent])
    compute_ratios = U.function([ob, ac], ratio)
    compute_kls = U.function([ob, ac], kloldnew)

    compute_rollout_old_prob = U.function([ob, ac], tf.reduce_mean(oldpi.pd.logp(ac)))
    compute_rollout_new_prob = U.function([ob, ac], tf.reduce_mean(pi.pd.logp(ac)))
    compute_rollout_new_prob_min = U.function([ob, ac], tf.reduce_min(pi.pd.logp(ac)))

    update_ops = {}
    update_placeholders = {}
    for v in pi.get_trainable_variables():
        update_placeholders[v.name] = tf.placeholder(v.dtype, shape=v.get_shape())
        update_ops[v.name] = v.assign(update_placeholders[v.name])

    # compute fisher information matrix
    dims = [int(np.prod(p.shape)) for p in pol_var_list]
    logprob_grad = U.flatgrad(tf.reduce_mean(pi.pd.logp(ac)), pol_var_list)
    compute_logprob_grad = U.function([ob, ac], logprob_grad)

    U.initialize()

    adam.sync()

    if init_policy_params is not None:
        cur_scope = pi.get_variables()[0].name[0:pi.get_variables()[0].name.find('/')]
        orig_scope = list(init_policy_params.keys())[0][0:list(init_policy_params.keys())[0].find('/')]
        print(cur_scope, orig_scope)
        for i in range(len(pi.get_variables())):
            if pi.get_variables()[i].name.replace(cur_scope, orig_scope, 1) in init_policy_params:
                assign_op = pi.get_variables()[i].assign(init_policy_params[pi.get_variables()[i].name.replace(cur_scope, orig_scope, 1)])
                tf.get_default_session().run(assign_op)
                assign_op = oldpi.get_variables()[i].assign(init_policy_params[pi.get_variables()[i].name.replace(cur_scope, orig_scope, 1)])
                tf.get_default_session().run(assign_op)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    prev_params = {}
    for v in var_list:
        if 'pol' in v.name:
            prev_params[v.name] = v.eval()

    optim_seg = None

    grad_scale = 1.0

    while True:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('begin')
            memory()
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError


        logger.log("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__()

        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        unstandardized_adv = np.copy(atarg)
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

        args = seg["ob"], seg["ac"], atarg
        fvpargs = [arr for arr in args]

        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))


        cur_clip_val = clip_param

        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)

        for epoch in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult,
                                            cur_clip_val)
                adam.update(g * grad_scale, optim_stepsize * cur_lrmult)
                losses.append(newlosses)

            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult, clip_param)
            losses.append(newlosses)
        meanlosses,_,_ = mpi_moments(losses, axis=0)

        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.log(fmt_row(13, meanlosses))
            for (lossval, name) in zipsame(meanlosses, loss_names):
                logger.record_tabular("loss_"+name, lossval)

        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.record_tabular("EpLenMean", np.mean(lenbuffer))
            logger.record_tabular("EpRewMean", np.mean(rewbuffer))
            logger.record_tabular("EpThisIter", len(lens))
            logger.record_tabular("PolVariance", repr(adam.getflat()[-env.action_space.shape[0]:]))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1


        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.record_tabular("EpisodesSoFar", episodes_so_far)
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            logger.record_tabular("TimeElapsed", time.time() - tstart)


        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('end')
            memory()



    return pi, np.mean(rewbuffer)

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
