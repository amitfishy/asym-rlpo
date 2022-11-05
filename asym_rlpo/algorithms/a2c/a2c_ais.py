import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from asym_rlpo.data import Episode

from .base_ais import A2C_ABC_AIS


class A2C_AIS(A2C_ABC_AIS):
    model_keys = {
        'agent': [
            'action_model',
            'observation_model',
            'reward_model',
            'history_model',
            'policy_model',
        ],
        'critic': [
            'action_model',
            'observation_model',
            'reward_model',
            'history_model',
            'vh_model',
        ],
        'ais_psi': [
            'action_model',
            'observation_model',
            'reward_model',
            'history_model',
            'h_ais_psi_base',
            'h_pred_rew',
            'h_pred_next_ais',
        ],
    }

    def compute_v_values(
        self, models: nn.ModuleDict, episode: Episode
    ) -> torch.Tensor:

        history_features = self.compute_history_features_AIS(
            models.critic.action_model,
            models.critic.observation_model,
            models.critic.reward_model,
            models.critic.history_model,
            episode.actions,
            episode.observations,
            episode.rewards,
        )
        vh_values = models.critic.vh_model(history_features).squeeze(-1)
        # print('history_features: ', history_features, history_features.shape)
        # print('vh_values: ', vh_values, vh_values.shape)
        return vh_values

    def compute_ais_losses(self, models: nn.ModuleDict, episode: Episode) -> Tuple[torch.Tensor, torch.Tensor]:
        history_features = self.compute_history_features_AIS(
            models.ais_psi.action_model,
            models.ais_psi.observation_model,
            models.ais_psi.reward_model,
            models.ais_psi.history_model,
            episode.actions,
            episode.observations,
            episode.rewards,
        )

        action_features = models.ais_psi.action_model(episode.actions)
        reward_features = models.ais_psi.reward_model(episode.rewards).unsqueeze(-1)

        # print('episode actions: ', episode.actions, episode.actions.shape)
        # print('episode action_features: ', action_features, action_features.shape)
        # print('episode obs: ', episode.observations, episode.observations.shape)
        # print('episode rewards: ', episode.rewards, episode.rewards.shape)
        # print('episode reward_feats: ', reward_features, reward_features.shape)


        psi_inputs = torch.cat([history_features, action_features], dim=-1)
        # print('psi_inputs: ', psi_inputs, psi_inputs.shape)
        psi_inputs = models.ais_psi.h_ais_psi_base(psi_inputs)
        pred_rew = models.ais_psi.h_pred_rew(psi_inputs)
        pred_next_ais = models.ais_psi.h_pred_next_ais(psi_inputs)

        # print('pred_rew: ', pred_rew, pred_rew.shape)
        # print('pred_next_ais: ', pred_next_ais, pred_next_ais.shape)
        # print('history_features: ', history_features, history_features.shape)
        next_rew_loss = F.mse_loss(reward_features, pred_rew)
        next_ais_loss = torch.mean(2*torch.norm(pred_next_ais[:-1], p=2, dim=-1)**2 - 4*torch.sum(pred_next_ais[:-1]*history_features[1:], dim=-1))

        # print('losses: ', next_rew_loss, next_ais_loss)
        # exit()
        return next_rew_loss, next_ais_loss

    def ais_loss(
        self,
        episode: Episode,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        next_rew_loss, next_ais_loss = self.compute_ais_losses(self.models, episode)
        return next_rew_loss, next_ais_loss
