#!/usr/bin/env python
import argparse
import logging
import logging.config
import random
import signal
from dataclasses import asdict, dataclass
from typing import Dict, NamedTuple

import numpy as np
import torch
import torch.nn as nn
import wandb
from gym_gridverse.rng import reset_gv_rng

from asym_rlpo.algorithms import A2C_ABC_AIS, make_a2c_algorithm_ais
from asym_rlpo.envs import Environment, LatentType, make_env
from asym_rlpo.evaluation import evaluate_returns
from asym_rlpo.q_estimators import q_estimator_factory
from asym_rlpo.sampling_ais import sample_episodes
from asym_rlpo.utils.aggregate import average
from asym_rlpo.utils.checkpointing import Serializer, load_data, save_data
from asym_rlpo.utils.config import get_config
from asym_rlpo.utils.device import get_device
from asym_rlpo.utils.dispenser import (
    DiscreteDispenser,
    DiscreteDispenserSerializer,
    TimeDispenser,
)
from asym_rlpo.utils.running_average import (
    InfiniteRunningAverage,
    RunningAverage,
    RunningAverageSerializer,
    WindowRunningAverage,
)
from asym_rlpo.utils.scheduling import make_schedule
from asym_rlpo.utils.timer import Timer, TimerSerializer
from asym_rlpo.utils.wandb_logger import WandbLogger, WandbLoggerSerializer

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # algorithm and environment
    parser.add_argument('env')
    parser.add_argument('algo', choices=['a2c', 'asym-a2c', 'asym-a2c-state'])

    # wandb arguments
    parser.add_argument('--wandb-entity', default='amitfishy')
    parser.add_argument('--wandb-project', default='asym-rlpo-test1')
    parser.add_argument('--wandb-group', default='test1')
    parser.add_argument('--wandb-tag', action='append', dest='wandb_tags')
    parser.add_argument('--wandb-offline', action='store_true')
    parser.add_argument('--wandb-dir', default='results')

    # wandb related
    parser.add_argument('--num-wandb-logs', type=int, default=100)
    
    parser.add_argument('--env-label', default=None)
    parser.add_argument('--algo-label', default=None)

    # truncated histories
    parser.add_argument('--truncated-histories', action='store_true')
    parser.add_argument('--truncated-histories-n', type=int, default=-1)

    # reproducibility
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--deterministic', action='store_true')

    # general
    parser.add_argument(
        '--max-simulation-timesteps', type=int, default=20_000
    )
    parser.add_argument('--max-episode-timesteps', type=int, default=1_000)
    parser.add_argument('--simulation-num-episodes', type=int, default=1)

    # evaluation
    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--evaluation-period', type=int, default=10)
    parser.add_argument('--evaluation-num-episodes', type=int, default=1)
    parser.add_argument('--evaluation-epsilon', type=float, default=1.0)

    # discounts
    parser.add_argument('--evaluation-discount', type=float, default=1.0)
    parser.add_argument('--training-discount', type=float, default=0.99)

    # target
    parser.add_argument('--target-update-period', type=int, default=10_000)

    # q-estimator
    parser.add_argument(
        '--q-estimator',
        choices=['mc', 'td0', 'td-n', 'td-lambda'],
        default='td0',
    )
    parser.add_argument('--q-estimator-n', type=int, default=None)
    parser.add_argument('--q-estimator-lambda', type=float, default=None)

    # negentropy schedule
    parser.add_argument('--negentropy-schedule', default='linear')
    # linear
    parser.add_argument('--negentropy-value-from', type=float, default=1.0)
    parser.add_argument('--negentropy-value-to', type=float, default=0.01)
    parser.add_argument('--negentropy-nsteps', type=int, default=2_000_000)
    # exponential
    parser.add_argument('--negentropy-halflife', type=int, default=500_000)

    #ais lambda weights
    parser.add_argument('--next-ais-lambda', type=float, default=1.0)
    parser.add_argument('--latent-lambda', type=float, default=1.0)

    # optimization
    parser.add_argument('--optim-lr-actor', type=float, default=1e-4)
    parser.add_argument('--optim-eps-actor', type=float, default=1e-4)
    parser.add_argument('--optim-lr-critic', type=float, default=1e-4)
    parser.add_argument('--optim-eps-critic', type=float, default=1e-4)
    parser.add_argument('--optim-lr-ais', type=float, default=2e-4)
    parser.add_argument('--optim-eps-ais', type=float, default=2e-4)
    parser.add_argument('--optim-max-norm', type=float, default=float('inf'))

    # device
    parser.add_argument('--device', default='auto')

    # misc
    parser.add_argument('--render', action='store_true')

    # temporary / development
    parser.add_argument('--hs-features-dim', type=int, default=0)
    parser.add_argument('--normalize-hs-features', action='store_true')

    # latent observation
    parser.add_argument('--latent-type', default='state')

    # gv models
    parser.add_argument('--gv-observation-representation', default='compact')
    parser.add_argument('--gv-state-representation', default='compact')

    parser.add_argument(
        '--gv-observation-grid-model-type',
        choices=['cnn', 'fc'],
        default='fc',
    )
    parser.add_argument(
        '--gv-observation-representation-layers',
        type=int,
        default=0,
    )

    parser.add_argument(
        '--gv-state-grid-model-type',
        choices=['cnn', 'fc'],
        default='cnn',
    )
    parser.add_argument(
        '--gv-state-representation-layers',
        type=int,
        default=0,
    )

    # checkpoint
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--checkpoint-period', type=int, default=200_000)

    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--model-filename', default=None)

    parser.add_argument('--save-modelseq', action='store_true')
    parser.add_argument('--modelseq-filename', default=None)

    args = parser.parse_args()
    args.env_label = args.env if args.env_label is None else args.env_label
    args.algo_label = args.algo if args.algo_label is None else args.algo_label
    args.wandb_mode = 'offline' if args.wandb_offline else None
    return args


@dataclass
class XStats:
    epoch: int = 0
    simulation_episodes: int = 0
    simulation_timesteps: int = 0
    optimizer_steps: int = 0
    training_episodes: int = 0
    training_timesteps: int = 0

    def asdict(self):
        return asdict(self)


class XStatsSerializer(Serializer[XStats]):
    def serialize(self, obj: XStats) -> Dict:
        return obj.asdict()

    def deserialize(self, obj: XStats, data: Dict):
        obj.epoch = data['epoch']
        obj.simulation_episodes = data['simulation_episodes']
        obj.simulation_timesteps = data['simulation_timesteps']
        obj.optimizer_steps = data['optimizer_steps']
        obj.training_episodes = data['training_episodes']
        obj.training_timesteps = data['training_timesteps']


class RunState(NamedTuple):
    env: Environment
    algo: A2C_ABC_AIS
    optimizer_actor: torch.optim.Optimizer
    optimizer_critic: torch.optim.Optimizer
    optimizer_ais: torch.optim.Optimizer
    wandb_logger: WandbLogger
    xstats: XStats
    timer: Timer
    running_averages: Dict[str, RunningAverage]
    dispensers: Dict[str, DiscreteDispenser]


class RunStateSerializer(Serializer[RunState]):
    def __init__(self):
        self.wandb_logger_serializer = WandbLoggerSerializer()
        self.xstats_serializer = XStatsSerializer()
        self.timer_serializer = TimerSerializer()
        self.running_average_serializer = RunningAverageSerializer()
        self.dispenser_serializer = DiscreteDispenserSerializer()

    def serialize(self, obj: RunState) -> Dict:
        return {
            'models': obj.algo.models.state_dict(),
            'target_models': obj.algo.target_models.state_dict(),
            'optimizer_actor': obj.optimizer_actor.state_dict(),
            'optimizer_critic': obj.optimizer_critic.state_dict(),
            'optimizer_ais': obj.optimizer_ais.state_dict(),
            'wandb_logger': self.wandb_logger_serializer.serialize(
                obj.wandb_logger
            ),
            'xstats': self.xstats_serializer.serialize(obj.xstats),
            'timer': self.timer_serializer.serialize(obj.timer),
            'running_averages': {
                k: self.running_average_serializer.serialize(v)
                for k, v in obj.running_averages.items()
            },
            'dispensers': {
                k: self.dispenser_serializer.serialize(v)
                for k, v in obj.dispensers.items()
            },
        }

    def deserialize(self, obj: RunState, data: Dict):
        obj.algo.models.load_state_dict(data['models'])
        obj.algo.target_models.load_state_dict(data['target_models'])
        obj.optimizer_actor.load_state_dict(data['optimizer_actor'])
        obj.optimizer_critic.load_state_dict(data['optimizer_critic'])
        obj.optimizer_ais.load_state_dict(data['optimizer_ais'])
        self.wandb_logger_serializer.deserialize(
            obj.wandb_logger,
            data['wandb_logger'],
        )
        self.xstats_serializer.deserialize(obj.xstats, data['xstats'])
        self.timer_serializer.deserialize(obj.timer, data['timer'])

        data_keys = data['running_averages'].keys()
        obj_keys = obj.running_averages.keys()
        if set(data_keys) != set(obj_keys):
            raise RuntimeError()
        for k, running_average in obj.running_averages.items():
            self.running_average_serializer.deserialize(
                running_average,
                data['running_averages'][k],
            )

        data_keys = data['dispensers'].keys()
        obj_keys = obj.dispensers.keys()
        if set(data_keys) != set(obj_keys):
            raise RuntimeError()
        for k, dispenser in obj.dispensers.items():
            self.dispenser_serializer.deserialize(
                dispenser,
                data['dispensers'][k],
            )


def setup() -> RunState:
    config = get_config()

    table = str.maketrans({'-': '_'})
    latent_type = LatentType[config.latent_type.upper().translate(table)]
    env = make_env(
        config.env,
        latent_type=latent_type,
        max_episode_timesteps=config.max_episode_timesteps,
    )

    # print(latent_type)
    # print(env)
    # print(env.observation_space)
    # print(env.latent_space)
    # print(env.action_space)
    # exit()

    algo = make_a2c_algorithm_ais(
        config.algo,
        env,
        truncated_histories=config.truncated_histories,
        truncated_histories_n=config.truncated_histories_n,
    )

    optimizer_ais = torch.optim.Adam(
        algo.models.parameters(),
        lr=config.optim_lr_ais,
        eps=config.optim_eps_ais,
    )
    optimizer_actor = torch.optim.Adam(
        algo.models.parameters(),
        lr=config.optim_lr_actor,
        eps=config.optim_eps_actor,
    )
    optimizer_critic = torch.optim.Adam(
        algo.models.parameters(),
        lr=config.optim_lr_critic,
        eps=config.optim_eps_critic,
    )

    wandb_logger = WandbLogger()

    xstats = XStats()
    timer = Timer()

    running_averages = {
        'avg_target_returns': InfiniteRunningAverage(),
        'avg_behavior_returns': InfiniteRunningAverage(),
        'avg100_behavior_returns': WindowRunningAverage(100),
    }

    wandb_log_period = config.max_simulation_timesteps // config.num_wandb_logs
    dispensers = {
        'target_update_dispenser': DiscreteDispenser(
            config.target_update_period
        ),
        'wandb_log_dispenser': DiscreteDispenser(wandb_log_period),
    }

    return RunState(
        env,
        algo,
        optimizer_actor,
        optimizer_critic,
        optimizer_ais,
        wandb_logger,
        xstats,
        timer,
        running_averages,
        dispensers,
    )


def save_checkpoint(runstate: RunState):
    """saves a checkpoint with the current runstate

    NOTE:  must be called within an active wandb.init context manager
    """
    config = get_config()

    if config.checkpoint is not None:
        assert wandb.run is not None

        logger.info('checkpointing...')
        runstate_serializer = RunStateSerializer()
        checkpoint = {
            'metadata': {
                'config': config._as_dict(),
                'wandb_id': wandb.run.id,
            },
            'data': runstate_serializer.serialize(runstate),
        }
        save_data(config.checkpoint, checkpoint)
        logger.info('checkpointing DONE')


def run(runstate: RunState) -> bool:
    config = get_config()
    logger.info('run %s %s', config.env_label, config.algo_label)

    (
        env,
        algo,
        optimizer_actor,
        optimizer_critic,
        optimizer_ais,
        wandb_logger,
        xstats,
        timer,
        running_averages,
        dispensers,
    ) = runstate

    avg_target_returns = running_averages['avg_target_returns']
    avg_behavior_returns = running_averages['avg_behavior_returns']
    avg100_behavior_returns = running_averages['avg100_behavior_returns']
    target_update_dispenser = dispensers['target_update_dispenser']
    wandb_log_dispenser = dispensers['wandb_log_dispenser']

    device = get_device(config.device)
    algo.to(device)

    # reproducibility
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        reset_gv_rng(config.seed)
        env.seed(config.seed)

    if config.deterministic:
        torch.use_deterministic_algorithms(True)

    # initialize return type
    q_estimator = q_estimator_factory(
        config.q_estimator,
        n=config.q_estimator_n,
        lambda_=config.q_estimator_lambda,
    )

    behavior_policy = algo.behavior_policy()
    evaluation_policy = algo.evaluation_policy()
    evaluation_policy.epsilon = config.evaluation_epsilon

    negentropy_schedule = make_schedule(
        config.negentropy_schedule,
        value_from=config.negentropy_value_from,
        value_to=config.negentropy_value_from/10.0,#config.negentropy_value_to,
        nsteps=config.negentropy_nsteps,
        halflife=config.negentropy_halflife,
    )
    weight_negentropy = negentropy_schedule(xstats.simulation_timesteps)

    # setup interrupt flag via signal
    interrupt = False

    def set_interrupt_flag():
        nonlocal interrupt
        interrupt = True
        logger.debug('signal received, setting interrupt=True')

    signal.signal(signal.SIGUSR1, lambda signal, frame: set_interrupt_flag())

    checkpoint_dispenser = TimeDispenser(config.checkpoint_period)
    checkpoint_dispenser.dispense()  # burn first dispense

    # main learning loop
    wandb.watch(algo.models)
    while xstats.simulation_timesteps < config.max_simulation_timesteps:
        if interrupt:
            break

        if checkpoint_dispenser.dispense():
            save_checkpoint(runstate)

        # evaluate policy
        algo.models.eval()

        if config.evaluation and xstats.epoch % config.evaluation_period == 0:
            if config.render:
                sample_episodes(
                    env,
                    evaluation_policy,
                    num_episodes=1,
                    render=True,
                )

            episodes = sample_episodes(
                env,
                evaluation_policy,
                num_episodes=config.evaluation_num_episodes,
            )
            
            mean_length = sum(map(len, episodes)) / len(episodes)
            returns = evaluate_returns(
                episodes, discount=config.evaluation_discount
            )
            avg_target_returns.extend(returns.tolist())
            logger.info(
                f'EVALUATE epoch {xstats.epoch}'
                f' simulation_timestep {xstats.simulation_timesteps}'
                f' return {returns.mean():.3f}'
            )
            wandb_logger.log(
                {
                    **xstats.asdict(),
                    'hours': timer.hours,
                    'diagnostics/target_mean_episode_length': mean_length,
                    'performance/target_mean_return': returns.mean(),
                    'performance/avg_target_mean_return': avg_target_returns.value(),
                }
            )

        episodes = sample_episodes(
            env,
            behavior_policy,
            num_episodes=config.simulation_num_episodes,
        )

        mean_length = sum(map(len, episodes)) / len(episodes)
        returns = evaluate_returns(
            episodes, discount=config.evaluation_discount
        )
        avg_behavior_returns.extend(returns.tolist())
        avg100_behavior_returns.extend(returns.tolist())

        wandb_log = wandb_log_dispenser.dispense(xstats.simulation_timesteps)

        if wandb_log:
            logger.info(
                'behavior log - simulation_step %d return %.3f',
                xstats.simulation_timesteps,
                returns.mean(),
            )
            wandb_logger.log(
                {
                    **xstats.asdict(),
                    'hours': timer.hours,
                    'diagnostics/behavior_mean_episode_length': mean_length,
                    'performance/behavior_mean_return': returns.mean(),
                    'performance/avg_behavior_mean_return': avg_behavior_returns.value(),
                    'performance/avg100_behavior_mean_return': avg100_behavior_returns.value(),
                }
            )

        # storing torch data directly
        episodes = [episode.torch().to(device) for episode in episodes]
        xstats.simulation_episodes += len(episodes)
        xstats.simulation_timesteps += sum(len(episode) for episode in episodes)
        weight_negentropy = negentropy_schedule(xstats.simulation_timesteps)

        # target model update
        if target_update_dispenser.dispense(xstats.simulation_timesteps):
            # Update the target network
            algo.target_models.load_state_dict(algo.models.state_dict())

        algo.models.train()
        # ais
        optimizer_ais.zero_grad()
        losses = [
            algo.ais_loss(episode)
            for episode in episodes
        ]

        #done with this one 
        # print('ais losses: ', losses)
        # print('exiting')
        # exit()

        if config.algo == 'a2c':
            next_rew_loss, next_ais_loss = zip(*losses)
            next_rew_losses, next_ais_losses = average(next_rew_loss), average(next_ais_loss)
            ais_loss = next_rew_losses + config.next_ais_lambda*next_ais_losses
        elif config.algo == 'asym-a2c':
            next_rew_loss, next_ais_loss, latent_loss = zip(*losses)
            next_rew_losses, next_ais_losses, latent_losses = average(next_rew_loss), average(next_ais_loss), average(latent_loss)
            ais_loss = next_rew_losses + config.next_ais_lambda*next_ais_losses + config.latent_lambda*latent_losses
        else:
            assert False, "Wrong algorithm choice."
        ais_loss.backward()
        ais_gradient_norm = nn.utils.clip_grad.clip_grad_norm_(
            algo.models.parameters(), max_norm=config.optim_max_norm
        )
        optimizer_ais.step()

        # critic
        optimizer_critic.zero_grad()
        losses = [
            algo.critic_loss(
                episode,
                discount=config.training_discount,
                q_estimator=q_estimator,
            )
            for episode in episodes
        ]
        # print('Critic Loss: ', losses)
        # exit()

        critic_loss = average(losses)
        critic_loss.backward()
        critic_gradient_norm = nn.utils.clip_grad.clip_grad_norm_(
            algo.models.parameters(), max_norm=config.optim_max_norm
        )
        optimizer_critic.step()

        # actor
        optimizer_actor.zero_grad()
        losses = [
            algo.actor_losses(
                episode,
                discount=config.training_discount,
                q_estimator=q_estimator,
            )
            for episode in episodes
        ]

        # print('actor losses: ', losses)
        # exit()

        actor_losses, negentropy_losses = zip(*losses)
        actor_loss = average(actor_losses)
        negentropy_loss = average(negentropy_losses)

        loss = actor_loss + weight_negentropy * negentropy_loss
        loss.backward()
        actor_gradient_norm = nn.utils.clip_grad.clip_grad_norm_(
            algo.models.parameters(), max_norm=config.optim_max_norm
        )
        optimizer_actor.step()

        if wandb_log:
            logger.info(
                'training log - simulation_step %d losses %.3f %.3f %.3f %.3f',
                xstats.simulation_timesteps,
                actor_loss,
                critic_loss,
                negentropy_loss,
                ais_loss,
            )
            wandb_logger.log(
                {
                    **xstats.asdict(),
                    'hours': timer.hours,
                    'training/losses/actor': actor_loss,
                    'training/losses/critic': critic_loss,
                    'training/losses/negentropy': negentropy_loss,
                    'training/losses/ais': ais_loss,
                    'training/weights/negentropy': weight_negentropy,
                    'training/gradient_norms/actor': actor_gradient_norm,
                    'training/gradient_norms/critic': critic_gradient_norm,
                }
            )

            if config.save_modelseq and config.modelseq_filename is not None:
                data = {
                    'metadata': {'config': config._as_dict()},
                    'data': {
                        'timestep': xstats.simulation_timesteps,
                        'model.state_dict': algo.models.state_dict(),
                    },
                }
                filename = config.modelseq_filename.format(
                    xstats.simulation_timesteps
                )
                save_data(filename, data)

        xstats.epoch += 1
        xstats.optimizer_steps += 1
        xstats.training_episodes += len(episodes)
        xstats.training_timesteps += sum(len(episode) for episode in episodes)

    done = not interrupt

    if done and config.save_model and config.model_filename is not None:
        data = {
            'metadata': {'config': config._as_dict()},
            'data': {'models.state_dict': algo.models.state_dict()},
        }
        save_data(config.model_filename, data)

    return done


def main():
    args = parse_args()
    wandb_kwargs = {
        'project': args.wandb_project,
        'entity': args.wandb_entity,
        'group': args.wandb_group,
        'tags': args.wandb_tags,
        'mode': args.wandb_mode,
        'dir': args.wandb_dir,
        'config': args,
    }

    try:
        checkpoint = load_data(args.checkpoint)
    except (TypeError, FileNotFoundError):
        checkpoint = None
    else:
        wandb_kwargs.update(
            {
                'resume': 'must',
                'id': checkpoint['metadata']['wandb_id'],
            }
        )

    with wandb.init(**wandb_kwargs):
        config = get_config()
        config._update(dict(wandb.config))

        logger.info('setup of runstate...')
        runstate = setup()
        logger.info('setup DONE')

        if checkpoint is not None:
            if checkpoint['metadata']['config'] != config._as_dict():
                raise RuntimeError(
                    'checkpoint config inconsistent with program config'
                )

            logger.debug('updating runstate from checkpoint')
            runstate_serializer = RunStateSerializer()
            runstate_serializer.deserialize(runstate, checkpoint['data'])

        logger.info('run...')
        done = run(runstate)
        logger.info('run DONE')

        save_checkpoint(runstate)

    return int(not done)


if __name__ == '__main__':
    logging.config.dictConfig(
        {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                },
            },
            'handlers': {
                'default_handler': {
                    'class': 'logging.StreamHandler',
                    'level': 'DEBUG',
                    'formatter': 'standard',
                    'stream': 'ext://sys.stdout',
                },
            },
            'loggers': {
                '': {
                    'handlers': ['default_handler'],
                    'level': 'DEBUG',
                    'propagate': False,
                }
            },
        }
    )

    raise SystemExit(main())
