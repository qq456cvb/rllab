import numpy as np


class GoalMemory(object):
    def __init__(self, max_size):
        self.max_size = int(max_size)

        self.goal = np.zeros((self.max_size, 2), dtype='float32')

        self._curr_size = 0
        self._curr_pos = 0

    def append(self, goal):
        """
        Args:
            exp (Experience):
        """
        if self._curr_size < self.max_size:
            self.goal[self._curr_pos] = goal
            self._curr_pos = (self._curr_pos + 1) % self.max_size
            self._curr_size += 1
        else:
            self.goal[self._curr_pos] = goal
            self._curr_pos = (self._curr_pos + 1) % self.max_size

    def sample(self, idx):
        idx = (self._curr_pos + idx) % self._curr_size
        return self.goal[idx]

    def __len__(self):
        return self._curr_size
