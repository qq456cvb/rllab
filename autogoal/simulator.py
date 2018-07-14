import multiprocessing as mp
import time
import os
import threading
from abc import abstractmethod, ABCMeta
from collections import defaultdict

import six
from six.moves import queue
import zmq
import numpy as np

from tensorpack.utils import logger
from tensorpack.utils.serialize import loads, dumps
from tensorpack.utils.concurrency import LoopThread, ensure_proc_terminate


__all__ = ['SimulatorProcess', 'SimulatorMaster',
           'SimulatorProcessStateExchange',
           'TransitionExperience']


class TransitionExperience(object):
    """ A transition of state, or experience"""

    def __init__(self, state, action, reward, **kwargs):
        """ kwargs: whatever other attribute you want to save"""
        self.state = state
        self.action = action
        self.reward = reward
        for k, v in six.iteritems(kwargs):
            setattr(self, k, v)


@six.add_metaclass(ABCMeta)
class SimulatorProcessBase(mp.Process):
    def __init__(self, idx):
        super(SimulatorProcessBase, self).__init__()
        self.idx = int(idx)
        self.name = u'simulator-{}'.format(self.idx)
        self.identity = self.name.encode('utf-8')

    @abstractmethod
    def _build_player(self):
        pass


class SimulatorProcessStateExchange(SimulatorProcessBase):
    """
    A process that simulates a player and communicates to master to
    send states and receive the next action
    """

    def __init__(self, idx, pipe_c2s, pipe_s2c):
        """
        Args:
            idx: idx of this process
            pipe_c2s, pipe_s2c (str): name of the pipe
        """
        super(SimulatorProcessStateExchange, self).__init__(idx)
        self.c2s = pipe_c2s
        self.s2c = pipe_s2c

    def run(self):
        player = self._build_player()
        context = zmq.Context()
        c2s_socket = context.socket(zmq.PUSH)
        c2s_socket.setsockopt(zmq.IDENTITY, self.identity)
        c2s_socket.set_hwm(10)
        c2s_socket.connect(self.c2s)

        s2c_socket = context.socket(zmq.DEALER)
        s2c_socket.setsockopt(zmq.IDENTITY, self.identity)
        s2c_socket.connect(self.s2c)

        player.reset()
        r, is_over = 0, False
        st = player.get_current_obs()
        while True:
            c2s_socket.send(dumps((self.identity, st, r, is_over)), copy=False)
            # action = player.action_space.sample()
            action = loads(s2c_socket.recv(copy=False).bytes)
            st, r, is_over, _ = player.step(action)
            if is_over:
                player.reset()



# compatibility
SimulatorProcess = SimulatorProcessStateExchange


class SimulatorMaster(threading.Thread):
    """ A base thread to communicate with all StateExchangeSimulatorProcess.
        It should produce action for each simulator, as well as
        defining callbacks when a transition or an episode is finished.
    """
    class ClientState(object):
        def __init__(self):
            self.memory = []    # list of Experience
            self.ident = None

    def __init__(self, pipe_c2s, pipe_s2c):
        super(SimulatorMaster, self).__init__()
        assert os.name != 'nt', "Doesn't support windows!"
        self.daemon = True
        self.name = 'SimulatorMaster'

        self.context = zmq.Context()

        self.c2s_socket = self.context.socket(zmq.PULL)
        self.c2s_socket.bind(pipe_c2s)
        self.c2s_socket.set_hwm(20)
        self.s2c_socket = self.context.socket(zmq.ROUTER)
        self.s2c_socket.bind(pipe_s2c)
        self.s2c_socket.set_hwm(20)

        # queueing messages to client
        self.send_queue = queue.Queue(maxsize=1000)

        def f():
            msg = self.send_queue.get()
            self.s2c_socket.send_multipart(msg, copy=False)
        self.send_thread = LoopThread(f)
        self.send_thread.daemon = True
        self.send_thread.start()

        # make sure socket get closed at the end
        def clean_context(soks, context):
            for s in soks:
                s.close()
            context.term()
        import atexit
        atexit.register(clean_context, [self.c2s_socket, self.s2c_socket], self.context)

    def run(self):
        self.clients = defaultdict(self.ClientState)
        try:
            while True:
                msg = loads(self.c2s_socket.recv(copy=False).bytes)
                ident, state, reward, isOver = msg
                client = self.clients[ident]
                if client.ident is None:
                    client.ident = ident
                # maybe check history and warn about dead client?
                self._process_msg(client, state, reward, isOver)
        except zmq.ContextTerminated:
            logger.info("[Simulator] Context was terminated.")

    @abstractmethod
    def _process_msg(self, client, state, reward, isOver):
        pass

    def __del__(self):
        self.context.destroy(linger=0)
