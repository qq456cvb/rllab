import numpy as np
import os
import uuid
import argparse

import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import sys
import os


import tensorflow.contrib.slim as slim
import tensorflow.contrib.rnn as rnn
from tensorpack import *
from tensorpack.utils.concurrency import ensure_proc_terminate, start_proc_mask_signal
from tensorpack.utils.serialize import dumps
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils import get_current_tower_context, optimizer
from rllab.envs.mujoco.maze.ant_maze_env import AntMazeEnv
from autogoal.simulator import *


import six
import numpy as np
from six.moves import queue

if six.PY3:
    from concurrent import futures

    CancelledError = futures.CancelledError
else:
    CancelledError = Exception


STATE_DIM = 41 + 2
ACTION_DIM = 8
LOCAL_TIME_MAX = 50


class Model(ModelDesc):
    def get_policy(self, state):
        with tf.variable_scope('policy'):
            with tf.variable_scope('mean'):
                l = state
                l = FullyConnected('fc1', l, 32, activation=tf.nn.tanh)
                l = FullyConnected('fc2', l, 32, activation=tf.nn.tanh)
                mean = FullyConnected('fc3', l, ACTION_DIM, activation=tf.identity)
            with tf.variable_scope('log_std'):
                l = state
                l = FullyConnected('fc1', l, 32, activation=tf.nn.tanh)
                l = FullyConnected('fc2', l, 32, activation=tf.nn.tanh)
                log_std = FullyConnected('fc3', l, ACTION_DIM, activation=tf.identity)
        return mean, log_std

    def get_value(self, state):
        with tf.variable_scope('value'):
            l = state
            l = FullyConnected('fc1', l, 32, activation=tf.nn.tanh)
            l = FullyConnected('fc2', l, 32, activation=tf.nn.tanh)
            l = FullyConnected('fc3', l, 1, activation=tf.identity)
        return tf.squeeze(l, 1)

    def inputs(self):
        return [
            tf.placeholder(tf.float32, [None, STATE_DIM], 'state_in'),
            tf.placeholder(tf.float32, [None, ACTION_DIM], 'action_in'),
            tf.placeholder(tf.float32, [None], 'discounted_return_in'),
            tf.placeholder(tf.float32, [None], 'history_action_prob_in'),
        ]

    def build_graph(self, state, action_target, discounted_return, history_action_prob):
        scope = 'A3C'
        with tf.variable_scope(scope):
            mean, log_std = self.get_policy(state)
            log_std = tf.maximum(log_std, tf.log(1e-6))

            # sample action
            z = tf.random_normal(tf.shape(mean))
            action = tf.identity(z * tf.exp(log_std) + mean, name='action')
            log_prob = - 0.5 * mean.shape.as_list()[-1] * tf.log(2 * np.pi) - 0.5 * tf.reduce_sum(tf.square(z), axis=-1)
            prob = tf.exp(log_prob, name='action_prob')
            value = self.get_value(state)
            # this is the value for each agent, not the global value
            value = tf.identity(value, name='pred_value')
            is_training = get_current_tower_context().is_training

            if not is_training:
                return

            z_target = (action_target - mean) / tf.exp(log_std)
            prob_target = - tf.reduce_sum(log_std, axis=-1) - 0.5 * tf.reduce_sum(tf.square(z_target), axis=-1) - \
                          0.5 * mean.shape.as_list()[-1] * tf.log(2 * np.pi)
            # using PPO
            ppo_epsilon = tf.get_variable('ppo_epsilon', shape=[], initializer=tf.constant_initializer(0.2),
                                           trainable=False)
            importance = prob_target / (history_action_prob + 1e-8)

            # advantage
            advantage = tf.subtract(discounted_return, tf.stop_gradient(value), name='advantage')

            policy_loss = -tf.minimum(importance * advantage, tf.clip_by_value(importance, 1 - ppo_epsilon, 1 + ppo_epsilon) * advantage)
            # TODO: Gaussian entropy loss
            # entropy_loss = pa * logpa
            value_loss = tf.square(value - discounted_return)

            # entropy_beta = tf.get_variable('entropy_beta', shape=[], initializer=tf.constant_initializer(0.005),
            #                                trainable=False)

            value_weight = tf.get_variable('value_weight', shape=[], initializer=tf.constant_initializer(0.2), trainable=False)

        l2_loss = regularize_cost_from_collection()
        pred_reward = tf.reduce_mean(value, name='predict_reward')
        true_reward = tf.reduce_mean(discounted_return, name='true_reward')
        advantage = tf.sqrt(tf.reduce_mean(tf.square(advantage)), name='rms_advantage')

        policy_loss = tf.reduce_mean(policy_loss, name='policy_loss')
        # entropy_loss = tf.reduce_mean(entropy_loss, name='entropy_loss')
        value_loss = tf.reduce_mean(value_loss, name='value_loss')
        self.loss = tf.add_n([policy_loss, value_weight * value_loss],
                        name='a3c_loss')

        add_moving_summary(policy_loss, value_loss, l2_loss, pred_reward, true_reward, advantage, self.loss)
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

        return self.loss

    def optimizer(self):
        lr = tf.get_variable('a3c_learning_rate', initializer=1e-4, trainable=False)
        opt = tf.train.AdamOptimizer(lr)
        gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.3))]
        opt = optimizer.apply_grad_processors(opt, gradprocs)
        return opt
