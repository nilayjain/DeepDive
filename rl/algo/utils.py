import sys
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
        if k % 10000 == 0:
            print(f'\rEpisode: {k}/{self.num_episodes}', end='')
            sys.stdout.flush()

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
