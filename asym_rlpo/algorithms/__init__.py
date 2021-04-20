import gym

from .dqn.fob_dqn import FOB_DQN
from .dqn.foe_dqn import FOE_DQN
from .dqn.poe_adqn import POE_ADQN
from .dqn.poe_dqn import POE_DQN


def make_algorithm(name, env: gym.Env):
    if name == 'fob-dqn':
        return FOB_DQN(env)

    if name == 'foe-dqn':
        return FOE_DQN(env)

    if name == 'poe-dqn':
        return POE_DQN(env)

    if name == 'poe-adqn':
        return POE_ADQN(env)

    raise ValueError(f'invalid algorithm name {name}')
