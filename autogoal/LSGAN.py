import glob
import numpy as np
import os
import argparse


from tensorpack import *
from tensorpack.utils.viz import stack_patches
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
import tensorflow as tf

from GAN import GANTrainer, RandomZData, GANModelDesc

from rllab.envs.base import Env
# from rllab.env.base import MDP
from rllab.misc.resolve import load_class


class Model(GANModelDesc):
    def __init__(self, batch, z_dim):
        self.batch = batch
        self.zdim = z_dim

    def inputs(self):
        return [tf.placeholder(tf.float32, (None, 2), 'input')]

    def generator(self, z):
        """ return an image generated from z"""
        l = FullyConnected('fc1', z, 128, activation=tf.nn.relu)
        l = FullyConnected('fc2', l, 128, activation=tf.nn.relu)
        l = FullyConnected('fc3', l, 2, activation=tf.identity)
        l = tf.identity(l, name='gen')
        return l

    @auto_reuse_variable_scope
    def discriminator(self, goal):
        """ return a (b, 1) logits"""
        l = FullyConnected('fc1', goal, 256, activation=tf.nn.relu)
        l = FullyConnected('fc2', l, 256, activation=tf.nn.relu)
        l = FullyConnected('fc3', l, 1, activation=tf.identity)
        return l

    def build_losses(self, output_real, output_fake):
        self.d_loss = 0.5 * tf.reduce_mean(tf.squared_difference(output_real, 1))\
                      + 0.5 * tf.reduce_mean(tf.squared_difference(output_fake, -1))
        self.d_loss = tf.identity(self.d_loss, name='d_loss')
        self.g_loss = 0.5 * tf.reduce_mean(tf.squared_difference(output_fake, 0))
        self.g_loss = tf.identity(self.g_loss, name='g_loss')
        add_moving_summary(self.g_loss, self.d_loss)

    def build_graph(self, goal_pos):

        z = tf.random_uniform([self.batch, self.zdim], -1, 1, name='z_train')
        z = tf.placeholder_with_default(z, [None, self.zdim], name='z')

        with argscope([FullyConnected],
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)):
            with tf.variable_scope('gen'):
                goal_gen = self.generator(z)
            tf.summary.histogram('generated-goal', goal_gen)
            with tf.variable_scope('discrim'):
                val_pos = self.discriminator(goal_pos)
                val_neg = self.discriminator(goal_gen)

        self.build_losses(val_pos, val_neg)
        self.collect_variables()

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=2e-4, trainable=False)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)


def get_args(default_batch=128, default_z_dim=4):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--sample', action='store_true', help='view generated examples')
    parser.add_argument('--data', help='a jpeg directory')
    parser.add_argument('--z-dim', help='hidden dimension', type=int, default=default_z_dim)
    parser.add_argument('--batch', help='batch size', type=int, default=default_batch)
    global args
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args


class DF(DataFlow):
    def __init__(self):
        self.env = load_class("mujoco.maze.ant_maze_env", Env, ["rllab", "envs"])()
        self.max_steps = 200

    def get_data(self):
        self.env.reset()
        # self.env.render()
        for i in range(self.max_steps):
            # print(env.action_space)
            action = self.env.action_space.sample()
            ob, _, done, _ = self.env.step(action)
            yield [ob[122:124]]
            # print(ob[122:124])
            # if i % 10 == 0:
            # self.env.render()
            # import time as ttime
            # time.sleep(timestep / speedup)
            if done:
                self.env.reset()


if __name__ == '__main__':
    args = get_args()
    M = Model(batch=args.batch, z_dim=args.z_dim)
    if args.sample:
        # sample(M, args.load)
        pass
    else:
        logger.auto_set_dir()
        GANTrainer(
            input=QueueInput(PrefetchDataZMQ(BatchData(DF(), args.batch), args.batch)),
            model=M).train_with_defaults(
            callbacks=[ModelSaver()],
            steps_per_epoch=300,
            max_epoch=200,
            session_init=SaverRestore(args.load) if args.load else None
        )