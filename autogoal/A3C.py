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
from .simulator import *


import six
import numpy as np
from six.moves import queue

if six.PY3:
    from concurrent import futures

    CancelledError = futures.CancelledError
else:
    CancelledError = Exception

GAMMA = 0.998
SIMULATOR_PROC = 50
STATE_DIM = 169
ACTION_DIM = 8
LOCAL_TIME_MAX = 10

# number of games per epoch roughly = STEPS_PER_EPOCH * BATCH_SIZE / 100
STEPS_PER_EPOCH = 100
BATCH_SIZE = 64
PREDICT_BATCH_SIZE = 32
PREDICTOR_THREAD_PER_GPU = 4
PREDICTOR_THREAD = None


def get_player():
    return AntMazeEnv()


class Model(ModelDesc):
    def get_policy(self, state):
        with tf.name_scope('policy'):
            l = state
            l = FullyConnected('fc1', l, 32, activation=tf.nn.tanh)
            l = FullyConnected('fc2', l, 32, activation=tf.nn.tanh)
            l = FullyConnected('fc3', l, ACTION_DIM, activation=tf.identity)

        return l

    def get_value(self, state):
        with tf.name_scope('value'):
            l = state
            l = FullyConnected('fc1', l, 32, activation=tf.nn.tanh)
            l = FullyConnected('fc2', l, 32, activation=tf.nn.tanh)
            l = FullyConnected('fc3', l, 1, activation=tf.identity)
        return tf.squeeze(l, 1)

    def inputs(self):
        return [
            tf.placeholder(tf.float32, [None, STATE_DIM], 'state_in'),
            tf.placeholder(tf.int32, [None], 'action_in'),
            tf.placeholder(tf.float32, [None], 'history_action_prob_in'),
            tf.placeholder(tf.float32, [None], 'discounted_return_in')
        ]

    def build_graph(self, state, action_target, history_action_prob, discounted_return):
        scope = 'A3C'
        with tf.name_scope(scope):
            logits = self.get_policy(state)
            prob = tf.nn.softmax(logits, name='action_prob')
            value = self.get_value(state)
            # this is the value for each agent, not the global value
            value = tf.identity(value, name='pred_value')
            is_training = get_current_tower_context().is_training

            if not is_training:
                return

            action_target_onehot = tf.one_hot(action_target, ACTION_DIM)

            # active mode
            logpa = tf.reduce_sum(action_target_onehot * tf.log(
                tf.clip_by_value(prob, 1e-7, 1 - 1e-7)), 1)

            # importance sampling
            pa = tf.reduce_sum(action_target_onehot * tf.clip_by_value(prob, 1e-7, 1 - 1e-7), 1)

            # using PPO
            ppo_epsilon = tf.get_variable('ppo_epsilon', shape=[], initializer=tf.constant_initializer(0.2),
                                           trainable=False)
            importance = pa / (history_action_prob + 1e-8)

            # advantage
            advantage = tf.subtract(discounted_return, tf.stop_gradient(value), name='advantage')

            policy_loss = -tf.minimum(importance * advantage, tf.clip_by_value(importance, 1 - ppo_epsilon, 1 + ppo_epsilon) * advantage)
            entropy_loss = pa * logpa
            value_loss = tf.square(value - discounted_return)

            entropy_beta = tf.get_variable('entropy_beta', shape=[], initializer=tf.constant_initializer(0.005),
                                           trainable=False)

            value_weight = tf.get_variable('value_weight', shape=[], initializer=tf.constant_initializer(0.2), trainable=False)

        l2_loss = regularize_cost_from_collection()
        pred_reward = tf.reduce_mean(value, name='predict_reward')
        true_reward = tf.reduce_mean(discounted_return, name='true_reward')
        advantage = tf.sqrt(tf.reduce_mean(tf.square(advantage)), name='rms_advantage')

        policy_loss = tf.reduce_mean(policy_loss, name='policy_loss')
        entropy_loss = tf.reduce_mean(entropy_loss, name='entropy_loss')
        value_loss = tf.reduce_mean(value_loss, name='value_loss')
        self.loss = tf.add_n([policy_loss, entropy_loss * entropy_beta, value_weight * value_loss],
                        name='a3c_loss')

        add_moving_summary(policy_loss, entropy_loss, value_loss, l2_loss, pred_reward, true_reward, advantage, self.loss,
                           importance)
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

        return self.loss

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-4, trainable=False)
        opt = tf.train.AdamOptimizer(lr)
        gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.3))]
        opt = optimizer.apply_grad_processors(opt, gradprocs)
        return opt


class MySimulatorWorker(SimulatorProcess):
    def _build_player(self):
        return AntMazeEnv()


class MySimulatorMaster(SimulatorMaster, Callback):
    def __init__(self, pipe_c2s, pipe_s2c, gpus):
        super(MySimulatorMaster, self).__init__(pipe_c2s, pipe_s2c)
        self.queue = queue.Queue(maxsize=BATCH_SIZE * 8 * 2)
        self._gpus = gpus

    def _setup_graph(self):
        # create predictors on the available predictor GPUs.
        nr_gpu = len(self._gpus)
        predictors = [self.trainer.get_predictor(
            ['state_in'],
            ['prob', 'pred_value'],
            self._gpus[k % nr_gpu])
            for k in range(PREDICTOR_THREAD)]
        self.async_predictor = MultiThreadAsyncPredictor(
            predictors, batch_size=PREDICT_BATCH_SIZE)

    def _before_train(self):
        self.async_predictor.start()

    def _on_state(self, state, client):
        """
        Launch forward prediction for the new state given by some client.
        """
        def cb(outputs):
            # logger.info('async predictor callback')
            try:
                distrib, value = outputs.result()
            except CancelledError:
                logger.info("Client {} cancelled.".format(client.ident))
                return
            assert np.all(np.isfinite(distrib)), distrib
            action = np.random.choice(len(distrib), p=distrib / distrib.sum())
            client.memory.append(TransitionExperience(
                state, action, reward=None, value=value, prob=distrib[action]))
            self.send_queue.put([client.ident, dumps((action))])
        self.async_predictor.put_task([state], cb)

    def _process_msg(self, client, state, reward, isOver):
        """
        Process a message sent from some client.
        """
        # in the first message, only state is valid,
        # reward&isOver should be discarde
        if len(client.memory > 0):
            # server receives one step later when it is finished
            client.memory[-1].reward = reward
            if isOver:
                self._parse_memory(0, client, True)
            else:
                if len(client.memory) == LOCAL_TIME_MAX + 1:
                    R = client.memory[-1].value
                    self._parse_memory(R, client, False)
        # feed state and return action
        self._on_state(state, client)

    def _parse_memory(self, init_r, client, isOver):
        # for each agent's memory
        for role_id in range(1, 4):
            mem = client.memory[role_id - 1]
            if not isOver:
                last = mem[-1]
                mem = mem[:-1]

            mem.reverse()
            R = float(init_r)
            for idx, k in enumerate(mem):
                R = np.clip(k.reward, -1, 1) + GAMMA * R
                self.queue.put([k.state, k.action, R, k.prob])

            if not isOver:
                client.memory = [last]
            else:
                client.memory = []