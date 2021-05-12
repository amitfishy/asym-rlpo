from typing import Union

import gym

from .a2c.asym_a2c import AsymA2C
from .a2c.base import A2C_Base
from .a2c.sym_a2c import SymA2C
from .dqn.adqn import ADQN, ADQN_Bootstrap
from .dqn.adqn_state import ADQN_State, ADQN_State_Bootstrap
from .dqn.base import DQN_Base
from .dqn.dqn import DQN
from .dqn.fob_dqn import FOB_DQN
from .dqn.foe_dqn import FOE_DQN


def make_algorithm(name, env: gym.Env) -> Union[DQN_Base, A2C_Base]:
    if name == 'fob-dqn':
        return FOB_DQN(env)

    if name == 'foe-dqn':
        return FOE_DQN(env)

    if name == 'dqn':
        return DQN(env)

    if name == 'adqn':
        return ADQN(env)

    if name == 'adqn-bootstrap':
        return ADQN_Bootstrap(env)

    if name == 'adqn-state':
        return ADQN_State(env)

    if name == 'adqn-state-bootstrap':
        return ADQN_State_Bootstrap(env)

    if name == 'sym-a2c':
        return SymA2C(env)

    if name == 'asym-a2c':
        return AsymA2C(env)

    raise ValueError(f'invalid algorithm name {name}')
