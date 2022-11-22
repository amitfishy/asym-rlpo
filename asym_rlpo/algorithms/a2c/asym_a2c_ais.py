import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from asym_rlpo.data import Episode

from .base_ais import A2C_ABC_AIS


class AsymA2C_AIS(A2C_ABC_AIS):
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
            'latent_model',
            'vhz_model',
        ],
        'ais_psi': [
            'action_model',
            'observation_model',
            'reward_model',
            'history_model',
            'latent_model',
            'hz_vanilla_ais_psi_base',
            'hz_asymac_ais_psi_base',
            'hz_pred_rew',
            'hz_pred_next_ais',
            'hz_pred_latent',
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
        latent_features = models.critic.latent_model(episode.latents)
        inputs = torch.cat([history_features, latent_features], dim=-1)
        vhz_values = models.critic.vhz_model(inputs).squeeze(-1)
        # print('history_features: ', history_features, history_features.shape)
        # print('latent_features: ', latent_features, latent_features.shape)
        # print('inputs: ', inputs, inputs.shape)
        # print('vhz_values: ', vhz_values, vhz_values.shape)
        return vhz_values

    def compute_ais_losses(self, models: nn.ModuleDict, episode: Episode) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        latent_features = models.ais_psi.latent_model(episode.latents)

        # print('episode actions: ', episode.actions, episode.actions.shape)
        # print('episode action_features: ', action_features, action_features.shape)
        # print('episode obs: ', episode.observations, episode.observations.shape)
        # print('episode rewards: ', episode.rewards, episode.rewards.shape)
        # print('episode reward_feats: ', reward_features, reward_features.shape)
        # print('episode latents: ', episode.latents, episode.latents.shape)
        # print('episode latent_feats: ', latent_features, latent_features.shape)

        psi_inputs = torch.cat([latent_features, history_features, action_features], dim=-1)
        # print('psi_inputs: ', psi_inputs, psi_inputs.shape)
        psi_inputs = models.ais_psi.hz_vanilla_ais_psi_base(psi_inputs)
        pred_rew = models.ais_psi.hz_pred_rew(psi_inputs)
        pred_next_ais = models.ais_psi.hz_pred_next_ais(psi_inputs)

        # print('pred_rew: ', pred_rew, pred_rew.shape)
        # print('pred_next_ais: ', pred_next_ais, pred_next_ais.shape)
        # print('history_features: ', history_features, history_features.shape)

        latent_psi_inputs = models.ais_psi.hz_asymac_ais_psi_base(history_features)
        pred_latent = models.ais_psi.hz_pred_latent(latent_psi_inputs)

        # print('latent_psi_inputs: ', latent_psi_inputs, latent_psi_inputs.shape)
        # print('pred_latent: ', pred_latent, pred_latent.shape)

        next_rew_loss = F.mse_loss(reward_features, pred_rew)
        next_ais_loss = torch.tensor(0.0)
        if pred_next_ais.shape[0] > 1:
            next_ais_loss = torch.mean(2*torch.norm(pred_next_ais[:-1], p=2, dim=-1)**2 - 4*torch.sum(pred_next_ais[:-1]*history_features[1:], dim=-1))

        latent_loss = torch.mean(2*torch.norm(pred_latent, p=2, dim=-1)**2 - 4*torch.sum(pred_latent*latent_features, dim=-1))

        # print('loss: ', next_rew_loss, next_ais_loss, latent_loss)
        return next_rew_loss, next_ais_loss, latent_loss

    def ais_loss(
        self,
        episode: Episode,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        next_rew_loss, next_ais_loss, latent_loss = self.compute_ais_losses(self.models, episode)
        return next_rew_loss, next_ais_loss, latent_loss