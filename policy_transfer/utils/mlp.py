import tensorflow as tf
import numpy as np
import baselines.common.tf_util as U
from policy_transfer.policies.utils import *
from policy_transfer.utils.common import *

class MLP:
    def __init__(self, name, in_dim, out_dim, layers, activation=tf.nn.relu, last_activation=tf.nn.sigmoid,
                 dropout=0.0, net_name = 'genff'):
        self.name = name
        with tf.variable_scope(name):
            self.params = []
            self.dropout = dropout
            self.activation = activation
            self.last_activation = last_activation
            self.layers = layers
            self.in_dim = in_dim
            self.out_dim = out_dim

            input = U.get_placeholder(name=name+'_input', dtype=tf.float32, shape=[None, in_dim])
            self.is_training = U.get_placeholder(name='is_training', dtype=tf.bool, shape=())

            last_out_dim = in_dim
            for i in range(len(layers)):
                w, b = dense_params(last_out_dim, layers[i], net_name+"%i" % (i + 1), weight_init=U.normc_initializer(1.0))
                self.params.append([w, b])
                last_out_dim = layers[i]
            w, b = dense_params(last_out_dim, out_dim, net_name+"_out", weight_init=U.normc_initializer(0.01))
            self.params.append([w, b])

            last_out = input
            for i in range(len(layers)):
                last_out = activation(tf.matmul(last_out, self.params[i][0]) + self.params[i][1])
                if dropout > 0.0:
                    last_out = tf.layers.dropout(last_out, rate=dropout, training=self.is_training)
            self.output = tf.matmul(last_out, self.params[-1][0]) + self.params[-1][1]
            if last_activation is not None:
                self.logits = self.output
                self.output = last_activation(self.output)

            self.inputs = [input, self.is_training]

            self._predict = U.function(self.inputs, self.output)
            if last_activation is not None:
                self._prelast = U.function(self.inputs, self.logits)

            self.scope = tf.get_variable_scope().name

    def get_output_symbolic(self, input, is_training=False):
        last_out = input
        for i in range(len(self.layers)):
            last_out = self.activation(tf.matmul(last_out, self.params[i][0]) + self.params[i][1])
            if self.dropout > 0.0:
                last_out = tf.layers.dropout(last_out, rate=self.dropout, training=is_training)
        output = tf.matmul(last_out, self.params[-1][0]) + self.params[-1][1]
        if self.last_activation is not None:
            output = self.last_activation(output)
        return output

    def predict(self, input, use_dropout=False):
        if np.array(input).ndim < 2:
            pred = self._predict(input[None], use_dropout==True)
        else:
            pred = self._predict(input, use_dropout == True)
        return pred

    def pre_lastlayer(self, input, use_dropout=False):
        if input.ndim < 2:
            pred = self._prelast(input[None], use_dropout==True)
        else:
            pred = self._prelast(input, use_dropout == True)
        return pred

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_variable_dict(self):
        save_dict = {}
        variables = self.get_variables()
        for i in range(len(variables)):
            cur_val = variables[i].eval()
            save_dict[variables[i].name] = cur_val
        return save_dict

    def set_variable_from_dict(self, pm):
        assign_params(self, pm)
