from typing import Sequence

import gym
from asym_rlpo.data import Episode, EpisodeBuffer
from asym_rlpo.policies import Policy, RandomPolicy

from .base import Algorithm


class DQN(Algorithm):
    def __init__(self, env: gym.Env):
        super().__init__()
        self.env = env
        self.episode_buffer = EpisodeBuffer(maxlen=1_000_000)

    def process(self, episodes: Sequence[Episode]):
        self.episode_buffer.append_episodes(episodes)

        print(
            f'episode_buffer stats - '
            f'#interactions: {self.episode_buffer.num_interactions()}'
            f'\t#episodes: {self.episode_buffer.num_episodes()}'
        )

    def behavior_policy(self) -> Policy:
        # TODO implement and instantiate the epsilon-greedy policy
        return RandomPolicy(self.env.action_space)

    def target_policy(self) -> Policy:
        # TODO implement and instantiate the argmax policy
        return RandomPolicy(self.env.action_space)
