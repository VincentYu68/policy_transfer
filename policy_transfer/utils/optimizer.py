import tensorflow as tf
import numpy as np
import baselines.common.tf_util as U
from policy_transfer.policies.utils import *
from policy_transfer.utils.common import *
from baselines.common import Dataset

# optimizer for classifier
class ClassifierOptimizer:
    def __init__(self, model, updater):
        self.model = model
        self.updater = updater

        #### classifier optimization
        gt_state = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        input_placeholders = model.inputs + [gt_state]
        logits = model.logits
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=gt_state)

        self.lossandgrad = U.function(input_placeholders, [loss, U.flatgrad(loss, model.get_trainable_variables())])
        self.loss = U.function(input_placeholders, loss)

        #### input optimization
        target_value = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        input_placeholders = model.inputs + [target_value]


    def fit_data(self, training_X, training_Y, iter_num = 200, batch_size=64,
                 stepsize=0.001, save_model_callback = None):

        dataset = Dataset(dict(X=np.array(training_X), Y=np.array(training_Y)), shuffle=True)
        losses = []
        for iter in range(iter_num):
            loss_epoch = []
            for batch in dataset.iterate_once(batch_size):
                inputs = [batch["X"], True, batch["Y"]]
                loss, g = self.lossandgrad(*inputs)
                self.updater.update(g, stepsize)
                loss_epoch.append(loss)
            losses.append(np.mean(loss_epoch))
            if iter % 5 == 0:
                print('iter: ', iter, 'loss: ', np.mean(loss_epoch))
            if save_model_callback is not None:
                save_model_callback(self.model, self.model.name, iter)

        return losses

    def optimize_input(self, X, iter_num = 200):
        pass




# optimizer for regression
class RegressorOptimizer:
    def __init__(self, model, updater):
        self.model = model
        self.updater = updater

        #### classifier optimization
        self.gt_state = tf.placeholder(dtype=tf.float32, shape=[None, model.out_dim])

        self.input_placeholders = model.inputs + [self.gt_state]
        loss = tf.losses.mean_squared_error(labels=self.gt_state, predictions=model.output)

        self.lossandgrad = U.function(self.input_placeholders, [loss, U.flatgrad(loss, model.get_trainable_variables())])
        self.loss = U.function(self.input_placeholders, loss)

    def setup_custom_loss(self, closs):
        loss = closs(labels=self.gt_state, predictions=self.model.output)
        self.lossandgrad = U.function(self.input_placeholders, [loss, U.flatgrad(loss, self.model.get_trainable_variables())])
        self.loss = U.function(self.input_placeholders, loss)

    def fit_data(self, training_X, training_Y, iter_num = 200, batch_size=64,
                 stepsize=0.001, save_model_callback = None):

        dataset = Dataset(dict(X=np.array(training_X), Y=np.array(training_Y)), shuffle=True)
        losses = []
        for iter in range(iter_num):
            loss_epoch = []
            for batch in dataset.iterate_once(batch_size):
                inputs = [batch["X"], True, batch["Y"]]
                loss, g = self.lossandgrad(*inputs)
                self.updater.update(g, stepsize)
                loss_epoch.append(loss)
            losses.append(np.mean(loss_epoch))
            if iter % 5 == 0:
                print('iter: ', iter, 'loss: ', np.mean(loss_epoch))
            if save_model_callback is not None:
                save_model_callback(self.model, self.model.name, iter)

        return losses

    def optimize_input(self, X, iter_num = 200):
        pass