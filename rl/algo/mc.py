from collections import Callable, defaultdict
from typing import List, Tuple

import gym
import numpy as np
from gym import Env

from rl.algo.utils import EpisodeUtils
from rl.algo.utils import choose, random_policy, epsilon_greedy_policy


class MonteCarlo:

    def __init__(self,
                 env: Env = gym.make("Blackjack-v0"),
                 num_episodes: int = 500000,
                 discount_factor: float = 1.,
                 first_visit: bool = False):
        self.env: Env = env
        self.num_episodes: int = num_episodes
        self.gamma: float = discount_factor
        self.first_visit: bool = first_visit
        self.eu = EpisodeUtils(num_episodes)

    def prediction(self,
                   policy: Callable[[Tuple], np.ndarray]) -> defaultdict:
        """
        Evaluates the given policy and returns the action value function.
        :param policy: policy to be evaluated.
        :return: action value function, q.
        """
        q = defaultdict(lambda: np.zeros(self.env.action_space.n))
        counts = defaultdict(lambda: np.zeros(self.env.action_space.n))
        for k in range(1, self.num_episodes + 1):
            self.eu.print_episode_num(k)
            ep = self._generate_episode(policy)
            self._process_episode(counts, ep, q)
        return q

    def control(self):
        """
        Start with a random policy, and find the optimal policy and the optimal
        action value function.
        :return: pi_*, and q_* (optimal policy and optimal action value func.)
        """
        na = self.env.action_space.n  # num_actions
        q = defaultdict(lambda: np.zeros(na))
        counts = defaultdict(lambda: np.zeros(na))
        policy = epsilon_greedy_policy(na, q)
        for k in range(1, self.num_episodes + 1):
            self.eu.print_episode_num(k)
            epsilon = 1. / k
            ep = self._generate_episode(policy, epsilon)
            self._process_episode(counts, ep, q)
        return policy, q

    def _process_episode(self, counts, ep, q):
        ep.reverse()
        g = 0.
        for i, step in enumerate(ep):
            # state, action, reward
            s, a, r = step
            g = r + self.gamma * g
            if self.first_visit and self._is_present(ep[i + 1:], s, a):
                continue
            counts[s][a] += 1
            q[s][a] += (1. / counts[s][a]) * (g - q[s][a])
        self.eu.add_ep(len(ep), g)

    def weighted_importance_sampling(self):
        """
        behave according to a random policy, and find the optimal policy and
        the optimal action value function.
        :return: pi_*, and q_* (optimal policy and optimal action value func.)
        """
        na = self.env.action_space.n  # num_actions
        q = defaultdict(lambda: np.zeros(na))
        c = defaultdict(lambda: np.zeros(na))
        target_policy = epsilon_greedy_policy(na, q)
        behaviour_policy = random_policy(na)
        for k in range(1, self.num_episodes + 1):
            self.eu.print_episode_num(k)
            ep = self._generate_episode(behaviour_policy)
            ep.reverse()
            g = 0.
            w = 1.
            for i, step in enumerate(ep):
                # state, action, reward
                s, a, r = step
                g = r + self.gamma * g
                c[s][a] += w
                q[s][a] += (w / c[s][a]) * (g - q[s][a])
                if a != target_policy(s, 0.):
                    break
                w *= 1. / (1. / self.env.action_space.n)
            self.eu.add_ep(len(ep), g)
        return target_policy, q

    def _generate_episode(self,
                          policy: Callable,
                          epsilon: float = 0.) -> List:
        obs = self.env.reset()
        steps = []
        while True:
            # sample action based on probabilities.
            probs = policy(obs, epsilon)
            action = choose(probs)
            new_obs, reward, done, _ = self.env.step(action)
            steps.append((obs, action, reward))
            obs = new_obs
            if done:
                break
        return steps

    @staticmethod
    def _is_present(episode: List, state: Tuple, action: int) -> bool:
        for step in episode:
            s, a, r = step
            if state == s and action == a:
                return True
        return False
