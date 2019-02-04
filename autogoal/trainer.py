import tensorflow as tf
import numpy as np
import argparse
from multiprocessing import Array
from tensorpack import (TowerTrainer,
                        ModelDescBase, DataFlow, StagingInput)
from tensorpack.tfutils.tower import TowerContext, TowerFuncWrapper
from tensorpack.utils.stats import StatCounter
from tensorpack.graph_builder import DataParallelBuilder, LeastLoadedDeviceSetter
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils.argtools import memoized
from autogoal.A3C import Model as A3CModel
from tensorpack import *
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient
from tensorpack.utils.concurrency import ensure_proc_terminate, start_proc_mask_signal
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.utils.serialize import dumps

from autogoal.simulator import SimulatorMaster, SimulatorProcess, TransitionExperience
from autogoal.LSGAN import Model as GANModel
from tensorpack import *
from rllab.envs.base import Env
# from rllab.env.base import MDP
from rllab.misc.resolve import load_class
from autogoal.GAN import GANModelDesc
from rllab.envs.mujoco.maze.ant_maze_env import AntMazeEnv
from autogoal.memory import GoalMemory
from queue import Queue
from scipy.spatial.distance import cdist
import uuid
import sys
import queue
import six


if six.PY3:
    from concurrent import futures
    CancelledError = futures.CancelledError
else:
    CancelledError = Exception

EPSILON = 0.5
GAMMA = 0.998
SIMULATOR_PROC = 10
BATCH_SIZE = 500
PREDICTOR_THREAD_PER_GPU = 4
PREDICTOR_THREAD = None
PREDICT_BATCH_SIZE = 32
MAX_STEP = 500
MEMORY_SIZE = 1000
GOAL_SAMPLE = 100
LABEL_EVAL = 10
R_MIN = 0.1
R_MAX = 0.9
goals = Array('f', [0] * (GOAL_SAMPLE * 3))


gan_df_queue = Queue()


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
        self.tower_func = TowerFuncWrapper(lambda *x: [gan_model.build_graph(*x[:len(gan_inputs_desc)]), a3c_model.build_graph(*x[len(gan_inputs_desc):])], gan_inputs_desc + a3c_inputs_desc)
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

        self.generator = self.get_predictor(['z'], ['gen/gen'])
        self.memory = GoalMemory(MEMORY_SIZE)
        self.env = WrappedEnv()
        def reset(s, goal):
            s.super().reset()
            s.goal = goal
        self.env.reset = reset

    def run_step(self):
        # sample noise
        z = np.random.normal(size=(GOAL_SAMPLE, 4))

        # generate goals
        gz = self.generator(z)[0]
        if len(self.memory) > 0:
            idx = np.random.randint(len(self.memory), size=(GOAL_SAMPLE // 2,))
            real_goals = np.stack([self.memory.sample(i) for i in idx], axis=0)
            real_labels = np.zeros([real_goals.shape[0]])
            sampled_goals = np.concatenate([gz, real_goals], axis=0)
        else:
            real_goals = np.zeros([0, 2], np.float32)
            real_labels = np.zeros([0], np.int32)
            sampled_goals = np.concatenate([gz, gz[:GOAL_SAMPLE // 2]], axis=0)

        # label the goals
        if len(self.memory) > 0:
            for i in real_goals.shape[0]:
                stat = StatCounter()
                for _ in range(LABEL_EVAL):
                    self.env.reset(real_goals[i])
                    done = False
                    r = 0
                    while not done:
                        _, r, done, _ = self.env.step(self.env.action_space.sample())
                    stat.feed(r)
                if R_MIN < stat.average < R_MAX:
                    real_labels[i] = 1

        gan_df_queue.put([z, real_goals, real_labels])

        goals[:] = sampled_goals.reshape(-1)

        # update policy, batch size 500
        for i in range(5):
            for j in range(1):
                logger.info('policy iter {},{}'.format(i, j))
                self.hooked_sess.run(self.a3c_min)

        # train GAN, batch size 128
        for i in range(2):
            logger.info('gan iter {}'.format(i))
            self.hooked_sess.run(self.d_min)
            self.hooked_sess.run(self.g_min)

        # insert memory
        if len(self.memory) > 0:
            dist = np.min(cdist(gz, self.memory.goal[:len(self.memory)], 'euclidean'), -1)
            for i in range(gz.shape[0]):
                if dist[i] > EPSILON:
                    self.memory.append(gz[i])
        else:
            for i in range(gz.shape[0]):
                self.memory.append(gz[i])


def get_args(default_batch=128, default_z_dim=4):
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load model')
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--z-dim', help='hidden dimension', type=int, default=default_z_dim)
    parser.add_argument('--batch', help='batch size', type=int, default=default_batch)
    global args
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args


class WrappedEnv(AntMazeEnv):
    def __init__(self):
        super().__init__()
        self.crt_step = 0
        self.goal = (0, 0)

    def reset(self):
        super().reset()
        self.crt_step = 0
        idx = np.random.randint(GOAL_SAMPLE * 3 // 2)
        self.goal = goals[idx * 2:idx * 2 + 2]
        ob = super().get_current_obs()
        return np.concatenate([ob[:29], ob[113:125], self.goal])

    def step(self, action):
        ob, _, done, info = super().step(action)
        st = np.concatenate([ob[:29], ob[113:125], self.goal])
        done = False
        if np.linalg.norm(self.wrapped_env.get_body_com("torso")[:2] - self.goal) < EPSILON:
            r = 1
            done = True
        else:
            r = 0
            if self.crt_step >= MAX_STEP:
                done = True
        self.crt_step += 1
        return st, r, done, info


class GANDF(DataFlow):
    def __init__(self):
        # self.env = load_class("mujoco.maze.ant_maze_env", Env, ["rllab", "envs"])()
        self.env = WrappedEnv()

    def get_data(self):
        while True:
            yield gan_df_queue.get()
            # self.env.reset()
            # done = False
            # while not done:
            #     # print(env.action_space)
            #     action = self.env.action_space.sample()
            #     ob, _, done, _ = self.env.step(action)
            #     # cv2.imshow('mujoco', self.env.render())
            #     # cv2.waitKey()
            #     yield [ob[-2:]]


class MySimulatorWorker(SimulatorProcess):
    def _build_player(self):
        return WrappedEnv()


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
            ['A3C/action', 'A3C/action_prob', 'A3C/pred_value'],
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
                action, prob, value = outputs.result()
            except CancelledError:
                logger.info("Client {} cancelled.".format(client.ident))
                return
            assert np.all(np.isfinite(prob)), prob
            client.memory.append(TransitionExperience(
                state, action, reward=None, value=value, prob=prob))
            self.send_queue.put([client.ident, dumps((action))])
        self.async_predictor.put_task([state], cb)

    def _process_msg(self, client, state, reward, isOver):
        """
        Process a message sent from some client.
        """
        # in the first message, only state is valid,
        # reward&isOver should be discarde
        if len(client.memory) > 0:
            # server receives one step later when it is finished
            client.memory[-1].reward = reward
            if isOver:
                self._parse_memory(0, client, True)
            # else:
            #     if len(client.memory) == LOCAL_TIME_MAX + 1:
            #         R = client.memory[-1].value
            #         self._parse_memory(R, client, False)
        # feed state and return action
        self._on_state(state, client)

    def _parse_memory(self, init_r, client, isOver):
        mem = client.memory
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


if __name__ == '__main__':
    args = get_args()

    logger.auto_set_dir()
    gan_df = GANDF()
    gan_input = QueueInput(gan_df)
    gan_model = GANModel(batch=args.batch, z_dim=args.z_dim)

    # assign GPUs for training & inference
    num_gpu = get_num_gpu()
    assert num_gpu > 0
    if num_gpu > 1:
        # use half gpus for inference
        predict_tower = list(range(num_gpu))[-num_gpu // 2:]
    else:
        predict_tower = [0]
    PREDICTOR_THREAD = len(predict_tower) * PREDICTOR_THREAD_PER_GPU
    train_tower = list(range(num_gpu))[:-num_gpu // 2] or [0]
    logger.info("[Batch-A3C] Train on gpu {} and infer on gpu {}".format(
        ','.join(map(str, train_tower)), ','.join(map(str, predict_tower))))

    # setup simulator processes
    name_base = str(uuid.uuid1())[:6]
    prefix = '@' if sys.platform.startswith('linux') else ''
    namec2s = 'ipc://{}sim-c2s-{}'.format(prefix, name_base)
    names2c = 'ipc://{}sim-s2c-{}'.format(prefix, name_base)
    procs = [MySimulatorWorker(k, namec2s, names2c, goals) for k in range(SIMULATOR_PROC)]
    ensure_proc_terminate(procs)
    start_proc_mask_signal(procs)

    master = MySimulatorMaster(namec2s, names2c, predict_tower)
    a3c_input = QueueInput(BatchData(DataFromQueue(master.queue), BATCH_SIZE))
    a3c_model = A3CModel()
    AGTrainer(
        gan_input=gan_input,
        gan_model=gan_model,
        a3c_input=a3c_input,
        a3c_model=a3c_model).train_with_defaults(
        callbacks=[
            ModelSaver(),
            master,
            StartProcOrThread(master)
        ],
        steps_per_epoch=300,
        max_epoch=200,
        session_init=SaverRestore(args.load) if args.load else None
    )
