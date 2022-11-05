import abc
from collections import deque
from typing import Callable, Deque, Optional

import torch

import asym_rlpo.generalized_torch as gtorch
from asym_rlpo.data import TorchObservation
from asym_rlpo.representations.base import Representation


def compute_input_features_AIS(
    action_model: Representation,
    observation_model: Representation,
    reward_model: Representation,
    action: Optional[torch.Tensor],
    observation: TorchObservation,
    reward: Optional[torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor:

    observation_features = observation_model(gtorch.to(observation, device))
    batch_shape = observation_features.shape[:-1]
    action_features = (
        torch.zeros(batch_shape + (action_model.dim,), device=device)
        if action is None
        else action_model(action.to(device))
    )
    reward_features = (
        torch.zeros(batch_shape + (1,), device=device)
        if reward is None
        else reward_model(reward.to(device)).unsqueeze(-1)
    )
    # print('action: ', action)
    # print('observation: ', observation)
    # print('reward: ', reward)
    # print('action_features: ', action_features)
    # print('observation_features: ', observation_features)
    # print('reward_features: ', reward_features)
    input_features = torch.cat([action_features, observation_features, reward_features], dim=-1)
    # print('input_features: ', input_features)

    # print('Embedded model test------>')
    # obs_embed = observation_model(torch.tensor([10], device=device))
    # print('Embed: ', obs_embed)
    return input_features


def compute_full_history_features_AIS(
    action_model: Representation,
    observation_model: Representation,
    reward_model: Representation,
    history_model: Representation,
    actions: torch.Tensor,
    observations: TorchObservation,
    rewards: torch.Tensor,
) -> torch.Tensor:

    action_features = action_model(actions)
    action_features = action_features.roll(1, 0)
    action_features[0, :] = 0.0
    observation_features = observation_model(observations)
    reward_features = reward_model(rewards).unsqueeze(-1)
    reward_features = reward_features.roll(1, 0)
    reward_features[0, :] = 0.0

    inputs = torch.cat([action_features, observation_features, reward_features], dim=-1)
    history_features, _ = history_model(inputs.unsqueeze(0))
    history_features = history_features.squeeze(0)
    
    # print('actions: ', actions, actions.shape)
    # print('action_features: ', action_features, action_features.shape)
    # print('obs: ', observations, observations.shape)
    # print('obs_feats: ', observation_features, observation_features.shape)
    # print('rewards: ', rewards, rewards.shape)
    # print('reward_feats: ', reward_features, reward_features.shape)
    # print('inputs: ', inputs, inputs.shape)
    # print('hist feats: ', history_features, history_features.shape)

    return history_features


#not updated for now
def compute_truncated_history_features_AIS(
    action_model: Representation,
    observation_model: Representation,
    history_model: Representation,
    actions: torch.Tensor,
    observations: TorchObservation,
    *,
    n: int,
) -> torch.Tensor:
    raise NotImplementedError

    action_features = action_model(actions)
    action_features = action_features.roll(1, 0)
    action_features[0, :] = 0.0
    observation_features = observation_model(observations)

    inputs = torch.cat([action_features, observation_features], dim=-1)
    padding = torch.zeros_like(inputs[0].expand(n - 1, -1))
    inputs = torch.cat([padding, inputs], dim=0).unfold(0, n, 1)
    inputs = inputs.swapaxes(-2, -1)
    history_features, _ = history_model(inputs)
    history_features = history_features[:, -1]

    return history_features


def compute_history_features_AIS(
    action_model: Representation,
    observation_model: Representation,
    reward_model: Representation,
    history_model: Representation,
    actions: torch.Tensor,
    observations: TorchObservation,
    rewards: torch.Tensor,
    *,
    truncated: bool,
    n: int,
) -> torch.Tensor:

    return (
        compute_truncated_history_features_AIS(
            action_model,
            observation_model,
            history_model,
            actions,
            observations,
            n=n,
        )
        if truncated
        else compute_full_history_features_AIS(
            action_model,
            observation_model,
            reward_model,
            history_model,
            actions,
            observations,
            rewards,
        )
    )


class HistoryIntegratorAIS(metaclass=abc.ABCMeta):
    def __init__(
        self,
        action_model: Representation,
        observation_model: Representation,
        reward_model: Representation,
        history_model: Representation,
    ):
        self.action_model = action_model
        self.observation_model = observation_model
        self.reward_model = reward_model
        self.history_model = history_model

    def compute_input_features_AIS(
        self, action: Optional[torch.Tensor], observation: TorchObservation, reward: Optional[torch.Tensor]
    ) -> torch.Tensor:

        # the history model is the only one guaranteed to have parameters
        device = next(self.history_model.parameters()).device
        return compute_input_features_AIS(
            self.action_model,
            self.observation_model,
            self.reward_model,
            action,
            observation,
            reward,
            device=device,
        )

    @abc.abstractmethod
    def reset(self, observation):
        assert False

    @abc.abstractmethod
    def step(self, action, observation, reward):
        assert False

    @property
    @abc.abstractmethod
    def features(self) -> torch.Tensor:
        assert False


class FullHistoryIntegratorAIS(HistoryIntegratorAIS):
    def __init__(
        self,
        action_model: Representation,
        observation_model: Representation,
        reward_model: Representation,
        history_model: Representation,
    ):
        super().__init__(
            action_model,
            observation_model,
            reward_model,
            history_model,
        )
        self.__features: torch.Tensor
        self.__hidden: torch.Tensor

    def reset(self, observation):
        input_features = self.compute_input_features_AIS(
            None,
            gtorch.unsqueeze(observation, 0),
            None,
        ).unsqueeze(1)
        self.__features, self.__hidden = self.history_model(input_features)
        self.__features = self.__features.squeeze(0).squeeze(0)

    def step(self, action, observation, reward):
        input_features = self.compute_input_features_AIS(
            action.unsqueeze(0),
            gtorch.unsqueeze(observation, 0),
            reward.unsqueeze(0),
        ).unsqueeze(1)
        self.__features, self.__hidden = self.history_model(
            input_features, hidden=self.__hidden
        )
        self.__features = self.__features.squeeze(0).squeeze(0)

    @property
    def features(self) -> torch.Tensor:
        return self.__features


#not updated for now
class TruncatedHistoryIntegratorAIS(HistoryIntegratorAIS):
    def __init__(
        self,
        action_model: Representation,
        observation_model: Representation,
        history_model: Representation,
        *,
        n: int,
    ):
        raise NotImplementedError
        super().__init__(
            action_model,
            observation_model,
            history_model,
        )
        self.n = n
        self._input_features_deque: Deque[torch.Tensor] = deque(maxlen=n)

    def reset(self, observation):
        input_features = self.compute_input_features(
            None,
            gtorch.unsqueeze(observation, 0),
        ).squeeze(0)

        self._input_features_deque.clear()
        self._input_features_deque.extend(
            torch.zeros(input_features.size(-1)) for _ in range(self.n - 1)
        )

        self._input_features_deque.append(input_features)

    def step(self, action, observation):
        input_features = self.compute_input_features(
            action.unsqueeze(0),
            gtorch.unsqueeze(observation, 0),
        ).squeeze(0)
        self._input_features_deque.append(input_features)

    @property
    def features(self) -> torch.Tensor:
        assert len(self._input_features_deque) == self.n

        input_features = torch.stack(
            tuple(self._input_features_deque)
        ).unsqueeze(0)

        history_features, _ = self.history_model(input_features)
        history_features = history_features.squeeze(0)[-1]

        return history_features


def make_history_integrator_AIS(
    action_model: Representation,
    observation_model: Representation,
    reward_model: Representation,
    history_model: Representation,
    *,
    truncated_histories: bool,
    truncated_histories_n: int,
) -> HistoryIntegratorAIS:
    return (
        TruncatedHistoryIntegratorAIS(
            action_model,
            observation_model,
            history_model,
            n=truncated_histories_n,
        )
        if truncated_histories
        else FullHistoryIntegratorAIS(
            action_model,
            observation_model,
            reward_model,
            history_model,
        )
    )


HistoryIntegratorMakerAIS = Callable[
    [Representation, Representation, Representation, Representation],
    HistoryIntegratorAIS,
]
HistoryFeaturesComputerAIS = Callable[
    [
        Representation,
        Representation,
        Representation,
        Representation,
        torch.Tensor,
        TorchObservation,
        torch.Tensor,
    ],
    torch.Tensor,
]
