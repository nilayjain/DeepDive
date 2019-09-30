import sys
from collections import Callable, defaultdict

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


class EpisodeUtils:
    """
    Helper class useful for plotting and printing episode statistics.
    credit:
    https://github.com/dennybritz/reinforcement-learning/blob/master/lib/plotting.py
    """

    def __init__(self, num_episodes: int):
        self.num_episodes = num_episodes
        self.episode_lengths = []
        self.episode_rewards = []

    def print_episode_num(self, k: int):
        m = self.num_episodes / 100 if self.num_episodes >= 100 else 1
        if k % m == 0:
            print(f'\rEpisode: {k}/{self.num_episodes}', end='')
            sys.stdout.flush()

    def add_ep(self, ep_len, ep_reward):
        self.episode_lengths.append(ep_len)
        self.episode_rewards.append(ep_reward)

    def plot(self):
        # plot episode length over time
        self._ep_len()
        # plot episode reward over time. smooth over a window of 10 episodes.
        self._ep_reward()
        # plot episodes per time step.
        self._ep_per_ts()

    def _ep_len(self):
        fig = plt.figure(figsize=(10, 5))
        plt.plot(self.episode_lengths)
        plt.xlabel('Episode')
        plt.ylabel('Episode Length')
        plt.title('Episode length over time')
        plt.show(fig)

    def _ep_reward(self, win=10):
        fig = plt.figure(figsize=(10, 5))
        rewards_smoothed = pd.Series(self.episode_rewards).rolling(
            win, min_periods=win).mean()
        plt.plot(rewards_smoothed)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Reward over time')
        plt.show(fig)

    def _ep_per_ts(self):
        fig = plt.figure(figsize=(10, 5))
        plt.plot(np.cumsum(self.episode_lengths),
                 np.arange(len(self.episode_lengths)))
        plt.xlabel("Time Steps")
        plt.ylabel("Episode")
        plt.title("Episode per time step")
        plt.show(fig)


def choose(probs):
    return np.random.choice(np.arange(len(probs)), p=probs)


def random_policy(num_actions: int) -> Callable:
    actions = np.ones(num_actions) / num_actions

    def policy(obs, epsilon=0.):
        return actions

    return policy


def epsilon_greedy_policy(num_actions: int, q: defaultdict) -> Callable:
    def policy(obs, epsilon):
        actions = np.ones(num_actions) * (epsilon / num_actions)
        greedy_action = np.argmax(q[obs])
        actions[greedy_action] += 1. - epsilon
        return actions

    return policy


def double_policy(p1: Callable, p2: Callable) -> Callable:
    def policy(obs, epsilon):
        a1 = p1(obs, epsilon)
        a2 = p2(obs, epsilon)
        return (a1 + a2) / 2.

    return policy
