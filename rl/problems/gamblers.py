import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

class GamblersProblem:
    """
    Gamblers Problem from Sutton, Barto.
    """

    def __init__(self,
                 p_h: float = 0.55,
                 discount_factor: float = 1.,
                 tolerance: float = 0.0001):
        """
        :param p_h: probability that heads will come
        :param discount_factor: b/w [0, 1]
        :param tolerance: if value function changes by less than this we will
        assume convergence.
        """
        self.p_h = p_h
        self.gamma = discount_factor
        self.theta = tolerance
        self.nS = 101
        self.V = np.zeros(self.nS)
        self.policy = np.zeros_like(self.V)

    def value_iter(self) -> Tuple[np.ndarray, np.ndarray]:
        while True:
            delta = 0.
            for s in range(self.nS):
                actions = self._lookahead(s)
                v = np.max(actions)
                delta = max(delta, abs(v - self.V[s]))
                self.V[s] = v
            if delta < self.theta:
                break

        for s in range(self.nS):
            actions = self._lookahead(s)
            self.policy[s] = np.argmax(actions)
        return self.policy, self.V

    def plot(self) -> None:
        # plot value function
        n = self.nS - 1
        x = range(len(self.V))
        plt.xlabel('capital')
        plt.ylabel('value function')
        plt.title('value function vs capital')
        plt.plot(x[:n], self.V[:n])
        plt.show()

        # plot policy
        plt.xlabel('capital')
        plt.ylabel('stake')
        plt.title('money staked vs capital')
        plt.plot(x[:n], self.policy[:n])
        plt.show()

    def _is_terminal(self, s: int) -> bool:
        return s == 0 or s == self.nS - 1

    def _lookahead(self, s: int) -> np.ndarray:
        actions = np.zeros(self.nS)
        for a in range(1, min(s, self.nS - 1 - s) + 1):
            n_p = s + a
            n_n = s - a
            reward = lambda x: 1. if x == self.nS - 1 else 0.
            actions[a] = (
                    self.p_h * (reward(n_p) + self.gamma * self.V[n_p]) +
                    (1. - self.p_h) * (reward(n_n) + self.gamma * self.V[n_n]))
        return actions
