from policy_transfer.utils.mlp import *
from policy_transfer.utils.optimizer import *
from policy_transfer.uposi.osi_env_wrapper import *
import numpy as np
import gym, joblib, tensorflow as tf
import policy_transfer.envs
from policy_transfer.policies.mirror_policy import *
from policy_transfer.policies.mlp_policy import *
from baselines import logger
from baselines.common import tf_util as U
from baselines.common.mpi_adam import MpiAdam
from mpi4py import MPI

def osi_train_callback(model, name, iter):
    params = model.get_variable_dict()
    joblib.dump(params, logger.get_dir()+'/osi_params.pkl', compress=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='DartHopperPT-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--name', help='name of experiments', type=str, default="")
    parser.add_argument('--OSI_hist', help='history step size', type=int, default=10)
    parser.add_argument('--policy_path', help='path to policy', type=str, default="")
    parser.add_argument('--dyn_params', action='append', type=int)

    parser.add_argument('--osi_iteration', help='number of iterations', type=int, default=6)
    parser.add_argument('--training_sample_num', help='number of training samples per iteration', type=int, default=20000)
    parser.add_argument('--action_noise', help='noise added to action', type=float, default=0.0)

    args = parser.parse_args()

    # extract information from args
    name = args.name
    seed = args.seed
    OSI_hist = args.OSI_hist
    policy_path = args.policy_path
    osi_iteration = args.osi_iteration
    training_sample_num = args.training_sample_num
    dyn_params = args.dyn_params


    # setup the environments
    # if use minitaur environment, set up differently
    if args.env == 'Minitaur':
        from pybullet_envs.minitaur.envs import minitaur_reactive_env
        from gym.wrappers import time_limit

        env_hist = time_limit.TimeLimit(minitaur_reactive_env.MinitaurReactiveEnv(render=False,
                                                                             accurate_motor_model_enabled=True,
                                                                             urdf_version='rainbow_dash_v0',
                                                                             include_obs_history=OSI_hist,
                                                                             include_act_history=0,
                                                                             train_UP=False),
                                                                             max_episode_steps=1000)
        env_up = time_limit.TimeLimit(minitaur_reactive_env.MinitaurReactiveEnv(render=False,
                                                                             accurate_motor_model_enabled=True,
                                                                             urdf_version='rainbow_dash_v0',
                                                                             include_obs_history=1,
                                                                             include_act_history=0,
                                                                             train_UP=True),
                                                                             max_episode_steps=1000)
    else:
        env_hist = gym.make(args.env)

        if env_hist.env.include_obs_history == 1 and env_hist.env.include_act_history == 0:
            from gym import spaces

            # modify observation space
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


    # configure things and load the learned universal policy
    config_name = 'data/osi_data/' + name + '_' + args.env + '_' + str(dyn_params)

    logger.configure(config_name, ['json', 'stdout'])

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
    updater = MpiAdam(osi.get_trainable_variables())
    optimizer = RegressorOptimizer(osi, updater)

    sess = tf.InteractiveSession()
    U.initialize()

    cur_scope = up_policy.get_variables()[0].name[0:up_policy.get_variables()[0].name.find('/')]
    orig_scope = list(policy_params.keys())[0][0:list(policy_params.keys())[0].find('/')]

    vars = up_policy.get_variables()
    for i in range(len(up_policy.get_variables())):
        assign_op = up_policy.get_variables()[i].assign(
            policy_params[up_policy.get_variables()[i].name.replace(cur_scope, orig_scope, 1)])
        sess.run(assign_op)

    osi_env = OSIEnvWrapper(env_hist, osi, OSI_hist, len(dyn_params))

    updater.sync()

    env_up.env.resample_MP = True
    env_hist.env.resample_MP = True

    env_up.seed(seed + MPI.COMM_WORLD.Get_rank())
    env_hist.seed(seed + MPI.COMM_WORLD.Get_rank())

    env_up.reset()
    env_hist.reset()

    input_data = []
    output_data = []
    for iter in range(osi_iteration):
        print('------------- Iter ', iter, ' ----------------')
        # collect samples
        env_to_use = env_up
        if iter > 0:
            env_to_use = osi_env

        lengths = []
        collected_data_size = 0
        while collected_data_size < training_sample_num:
            env_up.reset()
            env_hist.reset()
            # collect one trajectory
            o = env_to_use.reset()
            length = 0
            while True:
                true_dyn = env_to_use.env.param_manager.get_simulator_parameters()
                cur_state = env_to_use.env.state_vector()

                if iter == 0:
                    env_hist.env.set_state_vector(cur_state)
                    osi_input = env_hist.env._get_obs()
                else:
                    osi_input = env_to_use.env._get_obs()
                input_data.append(osi_input)
                output_data.append(true_dyn)

                act, _ = up_policy.act(True, o)
                o, r, d, _ = env_to_use.step(act + np.random.normal(0, args.action_noise, len(act)))
                length += 1
                collected_data_size += 1

                if d:
                    lengths.append(length)
                    break
        print('Average rollout length: ', np.mean(lengths))
        #print(input_data.shape, output_data.shape)
        # update osi model
        optimizer.fit_data(np.array(input_data), np.array(output_data), iter_num = 200, save_model_callback=osi_train_callback)
        params = osi.get_variable_dict()
        joblib.dump(params, logger.get_dir() + '/osi_params_'+str(iter)+'.pkl', compress=True)




