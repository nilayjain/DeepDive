from collections import Callable, defaultdict
from typing import List, Tuple

import gym
import numpy as np
from gym import Env


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

    def prediction(self,
                   policy: Callable[[Tuple], np.ndarray]) -> defaultdict:
        """
        Evaluates the given policy and returns the action value function.
        :param policy: policy to be evaluated.
        :return: action value function, q.
        """
        q = defaultdict(lambda: np.zeros(self.env.action_space.n))
        counts = defaultdict(lambda: np.zeros(self.env.action_space.n))
        for _ in range(self.num_episodes):
            ep = self._generate_episode(policy)
            ep.reverse()
            g = 0.
            for i, step in enumerate(ep):
                # state, action, reward
                s, a, r = step
                g = r + self.gamma * g
                if self.first_visit and self._is_present(ep[i + 1:], s, a):
                    continue
                counts[s][a] += 1
                q[s][a] = (q[s][a] + (1. / counts[s][a]) * (g - q[s][a]))
        return q

    def control(self):
        """
        Start with a random policy, and find the optimal policy and the optimal
        action value function.
        :return: pi_*, and q_* (optimal policy and optimal action value func.)
        """
        q = defaultdict(lambda: np.zeros(self.env.action_space.n))
        counts = defaultdict(lambda: np.zeros(self.env.action_space.n))
        policy = self._epsilon_greedy_policy(q)
        for k in range(1, self.num_episodes + 1):
            epsilon = 1. / k
            ep = self._generate_episode(policy, epsilon)
            ep.reverse()
            g = 0.
            for i, step in enumerate(ep):
                # state, action, reward
                s, a, r = step
                g = r + self.gamma * g
                if self.first_visit and self._is_present(ep[i + 1:], s, a):
                    continue
                counts[s][a] += 1
                q[s][a] = (q[s][a] + (1. / counts[s][a]) * (g - q[s][a]))
        return policy, q

    def weighted_importance_sampling(self):
        pass

    def _generate_episode(self,
                          policy: Callable,
                          epsilon: float = 0.) -> List:
        obs = self.env.reset()
        steps = []
        while True:
            # sample action based on probabilities.
            probs = policy(obs, epsilon)
            action = np.random.choice(np.arange(probs), p=probs)
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

    def _random_policy(self) -> Callable:
        nA = self.env.action_space.n
        actions = np.ones_like(nA) / nA

        def policy(obs, epsilon = 0.):
            return actions

        return policy

    def _epsilon_greedy_policy(self, q: defaultdict) -> Callable:

        def policy(obs, epsilon):
            actions = np.ones_like(self.env.action_space.n) * epsilon
            greedy_action = np.argmax(q[obs])
            actions[greedy_action] += 1. - epsilon
            return actions

        return policy
