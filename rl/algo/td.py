from collections import defaultdict, Callable

import gym
import numpy as np
from gym import Env

from rl.algo.utils import EpisodeUtils
from rl.algo.utils import epsilon_greedy_policy, choose, double_policy


class TemporalDifference:

    def __init__(self,
                 env: Env = gym.make("FrozenLake-v0"),
                 num_episodes: int = 5000,
                 discount_factor: float = 1.):
        self.env: Env = env
        self.num_episodes: int = num_episodes
        self.discount_factor: float = discount_factor
        self.eu: EpisodeUtils = EpisodeUtils(num_episodes)

    def sarsa(self, alpha: float = 0.1, epsilon: float = 0.1,
              expected_sarsa: bool = False, double_learning: bool = False,
              plot: bool = True):
        """
        Sarsa: On Policy TD Control
        :param alpha: learning rate.
        :param epsilon: probability of choosing random action for epsilon
        greedy policies
        :param expected_sarsa: use expected sarsa update
        :param double_learning: whether to use separate q functions for
        estimating value function and calculating maximizing action. (removes
        positive bias)
        :param plot: plot per episode statistics.
        :return: learned action value function, q.
        """
        na: int = not self.env.action_space.n  # num_actions
        policy, q1, q2 = self._init_q(double_learning, na)

        for k in range(1, self.num_episodes + 1):
            self.eu.print_episode_num(k)
            s = self.env.reset()
            el, er = 0, 0.
            a = choose(policy(s, epsilon))
            while True:
                n_s, reward, done, _ = self.env.step(a)
                n_a = choose(policy(n_s, epsilon))
                el += 1
                er += reward
                self._sarsa_update(q1, q2, s, n_s, a, n_a, reward,
                                   double_learning, expected_sarsa, alpha,
                                   policy, epsilon)
                if done:
                    self.eu.add_ep(el, er)
                    break
                s, a = n_s, n_a
        if plot:
            self.eu.plot()
        return q1

    def q_learning(self, alpha: float = 0.1, epsilon: float = 0.1,
                   double_learning: bool = False, plot: bool = True):
        """
        Q Learning: Off Policy TD Control
        :param alpha: step size
        :param epsilon: prob. of selecting random action for epsilon greedy
        policies
        :param double_learning: whether to use double learning
        :param plot: plot episode level statistics.
        :return:
        """
        na: int = not self.env.action_space.n  # num_actions
        policy, q1, q2 = self._init_q(double_learning, na)

        for k in range(1, self.num_episodes + 1):
            self.eu.print_episode_num(k)
            s = self.env.reset()
            el, er = 0, 0.
            while True:
                a = choose(policy(s, epsilon))
                n_s, reward, done, _ = self.env.step(a)
                el += 1
                er += reward
                self._q_learning_update(q1, q2, s, n_s, a, reward,
                                        double_learning, alpha)
                if done:
                    self.eu.add_ep(el, er)
                    break
                s = n_s
        if plot:
            self.eu.plot()
        return q1

    @staticmethod
    def _init_q(double_learning, na):
        q1 = defaultdict(lambda x: np.zeros(na))
        if double_learning:
            q2 = defaultdict(lambda x: np.zeros(na))
            p1 = epsilon_greedy_policy(na, q1)
            p2 = epsilon_greedy_policy(na, q2)
            policy = double_policy(p1, p2)
        else:
            q2 = None
            policy = epsilon_greedy_policy(na, q1)
        return policy, q1, q2

    def _sarsa_update(self,
                      q1: defaultdict,
                      q2: defaultdict,
                      s: int,
                      n_s: int,
                      a: int,
                      n_a: int,
                      reward: float,
                      double_learning: bool,
                      expected_sarsa: bool,
                      alpha: float,
                      policy: Callable,
                      epsilon: float) -> None:
        """
        Applies the sarsa update to action value function.
        :param q1: first action value function
        :param q2: second action value function (only req. in case of
        double learning)
        :param s: current state
        :param n_s: next state
        :param a: current action
        :param n_a: next action
        :param reward: reward from env on taking action a in state s.
        :param double_learning: set True to use double learning.
        :param expected_sarsa: set True for expected sarsa update
        :param alpha: step size
        :param policy: policy followed by the agent
        :param epsilon: for epsilon greedy policies, probability of taking a
        random action
        :return: None
        """

        def next_action_value(q: defaultdict, state: int) -> float:
            # get the expected value of state over the policy.
            probs = policy(state, epsilon)
            return probs @ q[state]

        if double_learning:
            p = np.random.sample()
            if expected_sarsa:
                # double learning expected sarsa update
                if p < 0.5:
                    q1[s][a] += alpha * (reward + self.discount_factor *
                                         next_action_value(q2, n_s) - q1[s][a])
                else:
                    q2[s][a] += alpha * (reward + self.discount_factor *
                                         next_action_value(q1, n_s) - q2[s][a])
            else:
                # double learning sarsa update
                if p < 0.5:
                    q1[s][a] += alpha * (reward + self.discount_factor *
                                         q2[n_s][n_a] - q1[s][a])
                else:
                    q2[s][a] += alpha * (reward + self.discount_factor *
                                         q1[n_s][n_a] - q2[s][a])
        else:
            if expected_sarsa:
                # expected sarsa update
                q1[s][a] += alpha * (reward + self.discount_factor *
                                     next_action_value(q1, n_s) - q1[s][a])
            else:
                # sarsa update
                q1[s][a] += alpha * (reward + self.discount_factor *
                                     q1[n_s][n_a] - q1[s][a])

    def _q_learning_update(self,
                           q1: defaultdict,
                           q2: defaultdict,
                           s: int,
                           n_s: int,
                           a: int,
                           reward: float,
                           double_learning: bool,
                           alpha: float) -> None:
        """
        Applies the Q Learning update to action value function.
        :param q1: first action value function
        :param q2: second action value function (only req. in case of
        double learning)
        :param s: current state
        :param n_s: next state
        :param a: current action
        :param reward: reward from env on taking action a in state s.
        :param double_learning: set True to use double learning.
        :param alpha: step size
        :return: None
        """
        if double_learning:
            p = np.random.sample()
            # double q learning
            if p < 0.5:
                n_a = np.argmax(q1[n_s])
                q1[s][a] += alpha * (reward + self.discount_factor *
                                     q2[n_s][n_a] - q1[s][a])
            else:
                n_a = np.argmax(q2[n_s])
                q2[s][a] += alpha * (reward + self.discount_factor *
                                     q1[n_s][n_a] - q2[s][a])
        else:
            # q learning
            n_a = np.argmax(q1[n_s])
            q1[s][a] += alpha * (reward + self.discount_factor *
                                 q1[n_s][n_a] - q1[s][a])
