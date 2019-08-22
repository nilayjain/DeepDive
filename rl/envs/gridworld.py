import io
import sys
from typing import List, Tuple, Union, Optional, Dict

import numpy as np
from gym.envs.toy_text.discrete import DiscreteEnv
from numpy.core._multiarray_umath import nditer


class GridWorld(DiscreteEnv):
    up, down, left, right = 0, 1, 2, 3

    def __init__(self,
                 shape: Optional[Union[Tuple, List]] = None):
        shape = self._validate_shape(shape)
        self.shape: List[int] = list(shape)
        nS: int = np.prod(shape)
        nA: int = 4
        # uniform distribution of states.
        isd: np.ndarray = np.ones(nS) / nS
        grid: np.ndarray = np.arange(nS).reshape(shape=shape)

        # dict of dicts of lists, where
        # P[s][a] == [(probability, next_state, reward, done), ...]
        P: Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]] = dict()
        y_m, x_m = shape[0], shape[1]
        actions: List[int] = [self.up, self.down, self.left, self.right]
        is_terminal = lambda x: x == 0 or x == nS - 1

        # iterate over all states and fill P.
        it: nditer = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            y, x = it.multi_index
            s: int = it.index
            if is_terminal(s):
                # will be in same state forever with 0 reward.
                for a in actions:
                    P[s][a] = [(1., s, 0., True)]
            else:
                for a in actions:
                    if a is self.up:
                        ns: int = s - x_m if y > 0 else s
                    elif a is self.down:
                        ns: int = s + x_m if y < y_m - 1 else s
                    elif a is self.left:
                        ns: int = s - 1 if x > 0 else s
                    else:
                        ns: int = s + 1 if x < x_m - 1 else s

                    r: float = 0. if is_terminal(ns) else -1.
                    P[s][a] = [(1., s, r, is_terminal(ns))]
            it.iternext()
        super().__init__(nS, nA, P, isd)

    @staticmethod
    def _validate_shape(shape):
        if shape is None:
            shape = [4, 4]
        if not isinstance(shape, Tuple) and not isinstance(shape, List):
            raise TypeError('shape requires a tuple or a list as input')
        if not len(shape) == 2:
            raise ValueError('shape requires a tuple or a list of len = 2')
        return shape

    def render(self, mode='human'):
        outfile = io.StringIO() if mode == 'ansi' else sys.stdout
        grid: np.ndarray = np.arange(self.nS).reshape(self.shape)
        it: np.nditer = np.nditer(grid, flags=['multi_index'])
        x_m: int = self.shape[1]
        while not it.finished:
            y, x = it.multi_index
            s = it.index

            if self.s == s:
                out: str = " x "
            elif s == 0 or s == self.nS - 1:
                out: str = " T "
            else:
                out: str = " o "

            if x == 0:
                out = out.lstrip()
            if x == x_m - 1:
                out = out.rstrip()
                out += '\n'
            outfile.write(out)
            it.iternext()
