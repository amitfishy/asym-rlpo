from __future__ import annotations

import abc
import random
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from asym_rlpo.data import Episode
from asym_rlpo.features_ais import HistoryIntegratorAIS
from asym_rlpo.policies_ais import HistoryPolicyAIS
from asym_rlpo.q_estimators import Q_Estimator, td0_q_estimator

from ..base_ais import Algorithm_ABC_AIS


class A2C_ABC_AIS(Algorithm_ABC_AIS):
    def behavior_policy(self) -> ModelPolicyAIS:
        history_integrator_AIS = self.make_history_integrator_AIS(
            self.models.agent.action_model,
            self.models.agent.observation_model,
            self.models.agent.reward_model,
            self.models.agent.history_model,
        )
        return ModelPolicyAIS(
            history_integrator_AIS,
            self.models.agent.policy_model,
        )

    def evaluation_policy(self) -> EpsilonGreedyModelPolicyAIS:
        history_integrator_AIS = self.make_history_integrator_AIS(
            self.models.agent.action_model,
            self.models.agent.observation_model,
            self.models.agent.reward_model,
            self.models.agent.history_model,
        )
        return EpsilonGreedyModelPolicyAIS(
            history_integrator_AIS,
            self.models.agent.policy_model,
        )

    def compute_action_logits(
        self, models: nn.ModuleDict, episode: Episode
    ) -> torch.Tensor:

        history_features_AIS = self.compute_history_features_AIS(
            models.agent.action_model,
            models.agent.observation_model,
            models.agent.reward_model,
            models.agent.history_model,
            episode.actions,
            episode.observations,
            episode.rewards,
        )
        action_logits = models.agent.policy_model(history_features_AIS)
        return action_logits

    @abc.abstractmethod
    def compute_v_values(
        self, models: nn.ModuleDict, episode: Episode
    ) -> torch.Tensor:
        assert False

    @abc.abstractmethod
    def ais_loss(
        self,
        episode: Episode
    ):
        assert False

    def actor_losses(  # pylint: disable=too-many-locals
        self,
        episode: Episode,
        *,
        discount: float,
        q_estimator: Optional[Q_Estimator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if q_estimator is None:
            q_estimator = td0_q_estimator

        action_logits = self.compute_action_logits(self.models, episode)
        device = action_logits.device

        with torch.no_grad():
            v_values = self.compute_v_values(self.models, episode)
            q_values = q_estimator(episode.rewards, v_values, discount=discount)

        discounts = discount ** torch.arange(len(episode), device=device)
        action_nlls = -action_logits.gather(
            1, episode.actions.unsqueeze(-1)
        ).squeeze(-1)
        advantages = q_values.detach() - v_values.detach()
        actor_loss = (discounts * advantages * action_nlls).sum()

        action_dists = torch.distributions.Categorical(logits=action_logits)
        negentropy_loss = -action_dists.entropy().sum()

        return actor_loss, negentropy_loss

    def critic_loss(  # pylint: disable=too-many-locals
        self,
        episode: Episode,
        *,
        discount: float,
        q_estimator: Optional[Q_Estimator] = None,
    ) -> torch.Tensor:

        if q_estimator is None:
            q_estimator = td0_q_estimator

        v_values = self.compute_v_values(self.models, episode)

        # print('v_values: ', v_values, v_values.shape)

        with torch.no_grad():
            target_v_values = self.compute_v_values(self.target_models, episode)
            target_q_values = q_estimator(
                episode.rewards, target_v_values, discount=discount
            )
        # print('target_q_values: ', target_q_values, target_q_values.shape)
        critic_loss = F.mse_loss(v_values, target_q_values, reduction='sum')

        return critic_loss


# policies


# function which maps history features to action-logits
PolicyFunctionAIS = Callable[[torch.Tensor], torch.Tensor]


class ModelPolicyAIS(HistoryPolicyAIS):
    def __init__(
        self,
        history_integrator_AIS: HistoryIntegratorAIS,
        policy_function_AIS: PolicyFunctionAIS,
    ):
        super().__init__(history_integrator_AIS)
        self.policy_function_AIS = policy_function_AIS

    def sample_action(self):
        action_logits = self.policy_function_AIS(self.history_integrator_AIS.features)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        return action_dist.sample().item()


class EpsilonGreedyModelPolicyAIS(HistoryPolicyAIS):
    def __init__(
        self,
        history_integrator_AIS: HistoryIntegratorAIS,
        policy_function_AIS: PolicyFunctionAIS,
    ):
        super().__init__(history_integrator_AIS)
        self.policy_function_AIS = policy_function_AIS

    def sample_action(self):
        action_logits = self.policy_function_AIS(self.history_integrator_AIS.features)

        if random.random() < self.epsilon:
            action_dist = torch.distributions.Categorical(logits=action_logits)
            return action_dist.sample().item()
        return action_logits.argmax().item()
