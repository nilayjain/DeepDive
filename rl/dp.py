import numpy as np

from rl.envs.gridworld import GridWorld


class DynamicProgramming:

    def __init__(self):
        self.env = GridWorld()
        self.gamma = 1.
        self.theta = 0.00001

    def policy_evaluation(self, policy) -> np.ndarray:
        V: np.ndarray = np.zeros(self.env.nS)
        while True:
            delta: float = 0.
            for s in range(self.env.nS):
                v: float = 0.
                for a in range(self.env.nA):
                    for p, ns, reward, done in self.env.P[s][a]:
                        v += policy[s, a] * (reward + p * self.gamma * V[ns])
                delta = max(delta, abs(v - V[s]))
                V[s] = v
            if delta < self.theta:
                break
        return V

    def policy_iteration(self):
        pass

    def value_iteration(self):
        pass
