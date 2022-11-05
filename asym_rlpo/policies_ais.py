import abc

import gym

from asym_rlpo.features_ais import HistoryIntegratorAIS


class PolicyAIS(metaclass=abc.ABCMeta):
    def __init__(self):
        self.epsilon = 1.0

    @abc.abstractmethod
    def reset(self, observation):
        assert False

    @abc.abstractmethod
    def step(self, action, observation, reward):
        assert False

    @abc.abstractmethod
    def sample_action(self):
        assert False


class RandomPolicyAIS(PolicyAIS):
    def __init__(self, action_space: gym.Space):
        super().__init__()
        self.action_space = action_space

    def reset(self, observation):
        pass

    def step(self, action, observation, reward):
        pass

    def sample_action(self):
        return self.action_space.sample()


class HistoryPolicyAIS(PolicyAIS):
    def __init__(self, history_integrator_AIS: HistoryIntegratorAIS):
        super().__init__()
        self.history_integrator_AIS = history_integrator_AIS

    def reset(self, observation):
        self.history_integrator_AIS.reset(observation)

    def step(self, action, observation, reward):
        self.history_integrator_AIS.step(action, observation, reward)
