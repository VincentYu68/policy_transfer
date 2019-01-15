# visualize uposi policy and plot the evaluated model parameters
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from policy_transfer.utils.mlp import *
from policy_transfer.uposi.osi_env_wrapper import *
import numpy as np
import gym, joblib, tensorflow as tf
import policy_transfer.envs
from policy_transfer.policies.mirror_policy import *
from policy_transfer.policies.mlp_policy import *
from baselines import logger
from baselines.common import tf_util as U
import time

def osi_train_callback(model, name, iter):
    params = model.get_variable_dict()
    joblib.dump(params, logger.get_dir()+'/osi_params.pkl', compress=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='DartHopperPT-v1')
    parser.add_argument('--OSI_hist', help='history step size', type=int, default=10)
    parser.add_argument('--policy_path', help='path to policy', type=str, default="")
    parser.add_argument('--osi_path', help='path to osi net', type=str, default="")
    parser.add_argument('--dyn_params', action='append', type=int)
    parser.add_argument('--reset_mp_prob', help='probability of reset mp', type=float, default=0)

    args = parser.parse_args()

    # extract information from args
    np.random.seed(0)
    OSI_hist = args.OSI_hist
    policy_path = args.policy_path
    osi_path = args.osi_path
    dyn_params = args.dyn_params

    env_hist = gym.make(args.env)

    if env_hist.env.include_obs_history == 1 and env_hist.env.include_act_history == 0:
        from gym import spaces

        env_hist.env.include_obs_history = OSI_hist
        env_hist.env.include_act_history = OSI_hist
        obs_dim_base = env_hist.env.obs_dim
        env_hist.env.obs_dim = env_hist.env.include_obs_history * obs_dim_base
        env_hist.env.obs_dim += len(env_hist.env.control_bounds[0]) * env_hist.env.include_act_history

        high = np.inf * np.ones(env_hist.env.obs_dim)
        low = -high
        env_hist.env.observation_space = spaces.Box(low, high)
        env_hist.observation_space = spaces.Box(low, high)

    env_hist.env.param_manager.activated_param = dyn_params
    env_hist.env.param_manager.controllable_param = dyn_params

    if hasattr(env_hist.env, 'disableViewer'):
        env_hist.env.disableViewer = False
    env_hist.env.visualize = True

    env_up = gym.make(args.env)
    env_up.env.train_UP = True
    env_up.env.param_manager.activated_param = dyn_params
    env_up.env.param_manager.controllable_param = dyn_params
    env_up.env.obs_dim += len(dyn_params)

    high = np.inf * np.ones(env_up.env.obs_dim)
    low = -high
    env_up.env.observation_space = spaces.Box(low, high)
    env_up.observation_space = spaces.Box(low, high)

    if hasattr(env_up.env, 'obs_perm'):
        obpermapp = np.arange(len(env_up.env.obs_perm), len(env_up.env.obs_perm) + len(dyn_params))
        env_up.env.obs_perm = np.concatenate([env_up.env.obs_perm, obpermapp])

    def policy_fn(name, ob_space, ac_space):
        hid_size = 64
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=hid_size, num_hid_layers=3)

    def policy_mirror_fn(name, ob_space, ac_space):
        return MirrorPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                            hid_size=64, num_hid_layers=3, observation_permutation=env_up.env.obs_perm,
                            action_permutation=env_up.env.act_perm, soft_mirror=False)

    test_param_dict = locals()
    with open(logger.get_dir() + "/testinfo.txt", "w") as text_file:
        text_file.write(str(test_param_dict))

    if 'mirror' in policy_path:
        pol_fn = policy_mirror_fn
    else:
        pol_fn = policy_fn

    up_policy = pol_fn("pi_test", env_up.observation_space, env_up.action_space)
    policy_params = joblib.load(policy_path)

    # define osi model and optimizer
    osi = MLP(name='osi', in_dim=env_hist.observation_space.shape[0], out_dim=len(dyn_params), layers=[256, 128, 64],
              activation=tf.nn.relu, last_activation=None, dropout=0.1)

    sess = tf.InteractiveSession()
    U.initialize()

    cur_scope = up_policy.get_variables()[0].name[0:up_policy.get_variables()[0].name.find('/')]
    orig_scope = list(policy_params.keys())[0][0:list(policy_params.keys())[0].find('/')]

    vars = up_policy.get_variables()
    for i in range(len(up_policy.get_variables())):
        assign_op = up_policy.get_variables()[i].assign(
            policy_params[up_policy.get_variables()[i].name.replace(cur_scope, orig_scope, 1)])
        sess.run(assign_op)

    osi.set_variable_from_dict(joblib.load(osi_path))

    osi_env = OSIEnvWrapper(env_hist, osi, OSI_hist, len(dyn_params))

    env_hist.env.resample_MP = True

    env_hist.reset()

    while True:
        input('Press enter to start the next rollout...')
        o = osi_env.reset()

        true_mp = []
        pred_mp = []
        length = 0
        one_error = 0
        while True:
            act, _ = up_policy.act(False, o)
            o, r, d, _ = osi_env.step(act)
            true_mp.append(np.copy(osi_env.env.param_manager.get_simulator_parameters()))
            pred_mp.append(np.copy(o[-len(dyn_params):]))
            one_error += np.sum(np.square(true_mp[-1] - pred_mp[-1]))
            length += 1
            osi_env.render()
            time.sleep(0.01)

            if np.random.random() < args.reset_mp_prob:
                osi_env.env.param_manager.resample_parameters()

            if d:
                break
        print('error: ', one_error / length)
        pred_mp = np.array(pred_mp)
        true_mp = np.array(true_mp)
        plt.figure()
        for d in range(len(dyn_params)):
            plt.plot(true_mp[:, d], label=str(d))
        plt.gca().set_prop_cycle(None)
        for d in range(len(dyn_params)):
            plt.plot(pred_mp[:, d], '--')
        plt.legend()
        plt.show()




