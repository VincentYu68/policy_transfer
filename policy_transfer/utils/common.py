from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
import sys
import joblib
import tensorflow as tf
import numpy as np
import inspect
import functools


def run_one_traj(env, policy, stochastic=True, observation_app = None, render=False):
    obs = []
    acs = []
    states = []
    rewards = []
    rew = 0
    o = env.reset()
    if observation_app is not None:
        o = np.concatenate([o, observation_app])
    obs.append(o)
    if hasattr(env, 'state_vector'):
        states.append(env.state_vector())
    d = False
    if policy.recurrent:
        init_state = policy.get_initial_state()
    while not d:
        if policy.recurrent:
            ac, init_state = policy.act(stochastic, o, init_state)
        else:
            ac = policy.act(stochastic, o)[0]
        o, r, d, _ = env.step(ac)
        if observation_app is not None:
            o = np.concatenate([o, observation_app])

        if render:
            env.render()
        if hasattr(env, 'state_vector'):
            states.append(env.state_vector())
        if hasattr(env.env, 'use_qdqstate'):
            if not env.env.use_qdqstate:
                ac = env.pad_action(ac)
        acs.append(ac)
        obs.append(o)
        rew += r
        rewards.append(r)
    return {'obs':obs, 'acs':acs, 'states':states}, rewards, rew

# from https://stackoverflow.com/questions/37670886/gathering-columns-of-a-2d-tensor-in-tensorflow
def gather_cols(params, indices, name=None):
    """Gather columns of a 2D tensor.

    Args:
        params: A 2D tensor.
        indices: A 1D tensor. Must be one of the following types: ``int32``, ``int64``.
        name: A name for the operation (optional).

    Returns:
        A 2D Tensor. Has the same type as ``params``.
    """
    with tf.op_scope([params, indices], name, "gather_cols") as scope:
        # Check input
        params = tf.convert_to_tensor(params, name="params")
        indices = tf.convert_to_tensor(indices, name="indices")
        try:
            params.get_shape().assert_has_rank(2)
        except ValueError:
            raise ValueError('\'params\' must be 2D.')
        try:
            indices.get_shape().assert_has_rank(1)
        except ValueError:
            raise ValueError('\'params\' must be 1D.')

        # Define op
        p_shape = tf.shape(params)
        p_flat = tf.reshape(params, [-1])
        i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1],
                                       [-1, 1]) + indices, [-1])
        return tf.reshape(tf.gather(p_flat, i_flat),
                          [p_shape[0], -1])

def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper



def assign_params(model, params):
    cur_scope = model.get_variables()[0].name[
                0:model.get_variables()[0].name.find('/')]
    orig_scope = list(params.keys())[0][0:list(params.keys())[0].find('/')]
    for j in range(len(model.get_variables())):
        assign_op = model.get_variables()[j].assign(
            params[model.get_variables()[j].name.replace(cur_scope, orig_scope, 1)])
        tf.get_default_session().run(assign_op)