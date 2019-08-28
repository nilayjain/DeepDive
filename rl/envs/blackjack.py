import io
import sys
from typing import List

from gym import Env
from gym.spaces import Discrete, Tuple
from gym.utils import seeding


class Blackjack(Env):
    """
    Blackjack Env from sutton, barto.
    """

    def __init__(self,
                 natural: bool = False):
        self.action_space = Discrete(2)
        self.observation_space = Tuple((Discrete(32), Discrete(11),
                                        Discrete(2)))
        self.natural = natural
        # Ace (1) - 10, Jack, Queen, King.
        self.deck: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
        # random number generator
        self.rng, _ = self.seed()
        self.reset()

    def step(self, action: int):
        assert self.action_space.contains(action)
        if action:  # hit
            self.player.append(self._draw_card())
            if self._is_bust(self.player):
                done = True
                reward = -1.
            else:
                done = False
                reward = 0.
        else:  # stick
            done = True
            while self._sum_hand(self.dealer) < 17:
                self.dealer.append(self._draw_card())
            reward = (1. if self._score(self.player) > self._score(self.dealer)
                      else -1.)
        return self._obs(), reward, done, {}

    def reset(self):
        self.dealer = self._draw_hand()
        self.player = self._draw_hand()

        while self._sum_hand(self.player) < 12:
            self.player.append(self._draw_card())
        return self._obs()

    def render(self, mode='human'):
        outfile = io.StringIO() if mode == 'ansi' else sys.stdout
        outfile.write(f'player: {self.player}, dealer: {self.dealer}')

    def seed(self, seed=None):
        return seeding.np_random(seed)

    def _obs(self):
        return (self._sum_hand(self.player), self.dealer[0],
                self._usable_ace(self.player))

    def _draw_card(self) -> int:
        return self.rng.choice(self.deck)

    def _draw_hand(self) -> List[int]:
        return [self._draw_card(), self._draw_card()]

    @staticmethod
    def _usable_ace(hand: List[int]) -> bool:
        return 1 in hand and sum(hand) + 10 <= 21

    def _sum_hand(self, hand: List[int]) -> int:
        return 10 + sum(hand) if self._usable_ace(hand) else sum(hand)

    def _is_bust(self, hand: List[int]) -> bool:
        return self._sum_hand(hand) > 21

    def _score(self, hand: List[int]) -> int:
        return 0 if self._is_bust(hand) else self._sum_hand(hand)

    @staticmethod
    def _is_natural(hand: List[int]) -> bool:
        return sorted(hand) == [1, 10]
