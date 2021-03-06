from typing import Optional

import numpy as np
from gym.envs.toy_text.discrete import DiscreteEnv

from rl.envs.gridworld import GridWorld


class DynamicProgramming:

    def __init__(self,
                 env: Optional[DiscreteEnv] = None,
                 tolerance: Optional[float] = 0.00001,
                 discount_factor: Optional[float] = 1.):
        # all the dp algorithms can work with any DiscreteEnv.
        if env is not None:
            assert isinstance(env, DiscreteEnv), 'dp requires discrete env'
        else:
            env = GridWorld()
        self.env = env
        self.theta = tolerance
        self.gamma = discount_factor

    def policy_evaluation(self, policy) -> np.ndarray:
        """
        Evaluate a policy and return its value function.
        :param policy: policy to be evaluated.
        :return: value function v_pi
        """
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
        """
        Start with a random policy. Perform Generalized Policy Iteration (GPI)
        to find the optimal policy and the optimal value function.
        GPI: do policy evaluation followed by policy improvement (act greedily
        w.r.t the value function).
        :return: optimal policy pi_*, and value function V_*
        """
        policy = np.ones([self.env.nS, self.env.nA]) / 4.
        while True:
            policy_stable: bool = True
            V: np.ndarray = self.policy_evaluation(policy)
            new_policy = np.zeros_like(policy)
            for s in range(self.env.nS):
                old_action: int = np.argmax(policy[s])
                action_values = self._one_step_lookahead(s, V)
                new_action = np.argmax(action_values)
                new_policy[new_action] = 1
                if old_action != new_action:
                    policy_stable = False
            if policy_stable:
                return policy, V

    def value_iteration(self):
        """
        Use the bellman optimality equation to find optimal value function V_*
        :return: optimal policy pi_*, and value function V_*
        """
        V = np.zeros(self.env.nS)
        while True:
            delta: float = 0.
            for s in range(self.env.nS):
                av: np.ndarray = self._one_step_lookahead(s, V)
                v: float = np.max(av)  # max value in this state
                delta = max(delta, abs(v - V[s]))
                V[s] = v
            if delta < self.theta:
                break
        policy = np.zeros([self.env.nS, self.env.nA])
        for s in range(self.env.nS):
            av: np.ndarray = self._one_step_lookahead(s, V)
            policy[s, np.argmax(av)] = 1.
        return policy, V

    def _one_step_lookahead(self, s: int, V: np.ndarray) -> np.ndarray:
        """
        Calculate all action values from state s.
        :param s: state
        :return: action values.
        """
        av = np.zeros(self.env.nA)
        for a in range(self.env.nA):
            for p, ns, r, _ in self.env.P[s][a]:
                av[a] += p * (r + self.gamma * V[ns])
        return av
