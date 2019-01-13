import tensorflow as tf
import numpy as np
import baselines.common.tf_util as U

def dense(x, size, name, weight_init=None, bias=True):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    ret = tf.matmul(x, w)
    if bias:
        b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
        return ret + b
    else:
        return ret

def dense_params(in_size, out_size, name, weight_init=None):
    w = tf.get_variable(name + "/w", [in_size, out_size], initializer=weight_init)
    b = tf.get_variable(name + "/b", [out_size], initializer=tf.zeros_initializer())
    return w,b

class GenericModule(object):
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, in_dim, out_dim, hid_size, num_hid_layers, last_init_size = 0.01, name='ff'):
        # state_dim: dimension of input/output state from previous/root encoder
        self.params = []
        self.num_hid_layers = num_hid_layers

        self.intin_dim = in_dim

        last_out_dim = in_dim
        for i in range(num_hid_layers):
            w, b = dense_params(last_out_dim, hid_size, name+"%i"%(i+1), weight_init=U.normc_initializer(1.0))
            self.params.append([w,b])
            last_out_dim = hid_size
        w, b = dense_params(last_out_dim, out_dim, name+"_out", weight_init=U.normc_initializer(last_init_size))
        self.params.append([w, b])

    def get_output_tensor(self, input, activation=tf.nn.relu, out_activation=None):
        last_out = input
        for i in range(self.num_hid_layers):
            last_out = activation(tf.matmul(last_out, self.params[i][0]) + self.params[i][1])
        out = tf.matmul(last_out, self.params[-1][0]) + self.params[-1][1]
        if out_activation is not None:
            out = out_activation(out)
        return out

class GenericFF(object):
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, intin_dim, extin_dims, out_dim, hid_size, num_hid_layers):
        # state_dim: dimension of input/output state from previous/root encoder
        self.params = []
        self.num_hid_layers = num_hid_layers

        self.intin_dim = intin_dim
        self.extin_dims = extin_dims

        last_out_dim = intin_dim + np.sum(extin_dims)
        for i in range(num_hid_layers):
            w, b = dense_params(last_out_dim, hid_size, "genff%i"%(i+1), weight_init=U.normc_initializer(1.0))
            self.params.append([w,b])
            last_out_dim = hid_size
        w, b = dense_params(last_out_dim, out_dim, "genff_out", weight_init=U.normc_initializer(0.01))
        self.params.append([w, b])

    def get_output_tensor(self, state_in, input_in, activation=tf.nn.relu):
        if input_in is not None:
            last_out = tf.concat([state_in, input_in], axis=1)  # state_in
        else:
            last_out = state_in
        for i in range(self.num_hid_layers):
            last_out = activation(tf.matmul(last_out, self.params[i][0]) + self.params[i][1])
        out = tf.matmul(last_out, self.params[-1][0]) + self.params[-1][1]
        return out

class CosineModule(object):
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, in_dim, out_dim, hid_size, num_hid_layers, last_init_size = 0.01):
        # state_dim: dimension of input/output state from previous/root encoder
        self.params = []
        self.num_hid_layers = num_hid_layers

        self.intin_dim = in_dim-1

        last_out_dim = in_dim-1
        for i in range(num_hid_layers):
            w, b = dense_params(last_out_dim, hid_size, "ff%i"%(i+1), weight_init=U.normc_initializer(1.0))
            logmask = tf.get_variable(name="logmask%i"%(i+1), shape=[1],
                            initializer=tf.constant_initializer(-1.0))
            self.params.append([w,b, logmask])
            last_out_dim = hid_size
        w, b = dense_params(last_out_dim, out_dim, "ff_out", weight_init=U.normc_initializer(last_init_size))
        logmask = tf.get_variable(name="logmask_out", shape=[1],
                                  initializer=tf.constant_initializer(-1.0))
        self.params.append([w, b, logmask])

    def get_output_tensor(self, input, activation=tf.nn.relu, out_activation=None):
        last_out, time = tf.split(input, [self.intin_dim, 1], 1)

        for i in range(self.num_hid_layers):
            last_out = activation(tf.matmul(last_out, self.params[i][0]) + self.params[i][1] * tf.cos(time / tf.exp(self.params[i][2])))
        out = tf.matmul(last_out, self.params[-1][0]) + self.params[-1][1] * tf.cos(time / tf.exp(self.params[-1][2]))
        if out_activation is not None:
            out = out_activation(out)
        return out