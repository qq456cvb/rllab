import tensorflow as tf
import numpy as np
from tensorpack import (TowerTrainer,
                        ModelDescBase, DataFlow, StagingInput)
from tensorpack.tfutils.tower import TowerContext, TowerFuncWrapper
from tensorpack.graph_builder import DataParallelBuilder, LeastLoadedDeviceSetter
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils.argtools import memoized
from .GAN import GANModelDesc


class AGTrainer(TowerTrainer):
    def __init__(self, gan_input, a3c_input, gan_model, a3c_model):
        """
        Args:
            input (InputSource):
            model (GANModelDesc):
        """
        super(AGTrainer, self).__init__()
        assert isinstance(gan_model, GANModelDesc), gan_model
        gan_inputs_desc = gan_model.get_inputs_desc()
        a3c_inputs_desc = a3c_model.get_inputs_desc()
        self.register_callback(gan_input.setup(gan_inputs_desc) + a3c_input.setup(a3c_inputs_desc))

        """
        We need to set tower_func because it's a TowerTrainer,
        and only TowerTrainer supports automatic graph creation for inference during training.
        If we don't care about inference during training, using tower_func is
        not needed. Just calling model.build_graph directly is OK.
        """
        # Build the graph
        self.tower_func = TowerFuncWrapper(lambda x: [gan_model(x[:len(gan_inputs_desc)]), a3c_model(x[len(gan_inputs_desc):])], gan_inputs_desc + a3c_inputs_desc)
        with TowerContext('', is_training=True):
            self.tower_func(*(gan_input.get_input_tensors() + a3c_input.get_input_tensors()))

        gan_opt = gan_model.get_optimizer()
        with tf.name_scope('gan_optimize'):
            self.d_min = gan_opt.minimize(
                gan_model.d_loss, var_list=gan_model.d_vars, name='d_min')
            self.g_min = gan_opt.minimize(
                gan_model.g_loss, var_list=gan_model.g_vars, name='g_min')

        a3c_opt = a3c_model.get_optimizer()
        with tf.name_scope('a3c_optimize'):
            self.a3c_min = a3c_opt.minimize(
                a3c_model.loss, var_list=a3c_model.vars, name='a3c_min'
            )

    def run_step(self):
        self.hooked_sess.run(self.d_min)
        self.hooked_sess.run(self.g_min)
        self.hooked_sess.run(self.a3c_min)
