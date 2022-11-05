import functools

from asym_rlpo.envs import Environment
from asym_rlpo.features import compute_history_features, make_history_integrator
from asym_rlpo.features_ais import compute_history_features_AIS, make_history_integrator_AIS
from asym_rlpo.models import make_models, make_models_AIS

from .a2c.a2c import A2C
from .a2c.a2c_ais import A2C_AIS
from .a2c.asym_a2c import AsymA2C
from .a2c.asym_a2c_ais import AsymA2C_AIS
from .a2c.asym_a2c_state import AsymA2C_State
from .a2c.base import A2C_ABC
from .a2c.base_ais import A2C_ABC_AIS
from .dqn.adqn import ADQN, ADQN_Bootstrap
from .dqn.adqn_short import ADQN_Short
from .dqn.adqn_state import ADQN_State, ADQN_State_Bootstrap
from .dqn.base import DQN_ABC
from .dqn.dqn import DQN


def make_a2c_algorithm(
    name: str,
    env: Environment,
    *,
    truncated_histories: bool,
    truncated_histories_n: int,
) -> A2C_ABC:

    partial_make_history_integrator = functools.partial(
        make_history_integrator,
        truncated_histories=truncated_histories,
        truncated_histories_n=truncated_histories_n,
    )
    partial_compute_history_features = functools.partial(
        compute_history_features,
        truncated=truncated_histories,
        n=truncated_histories_n,
    )

    if name == 'a2c':
        algorithm_class = A2C
    elif name == 'asym-a2c':
        algorithm_class = AsymA2C
    elif name == 'asym-a2c-state':
        algorithm_class = AsymA2C_State
    else:
        raise ValueError(f'invalid algorithm name {name}')

    models = make_models(env, keys=algorithm_class.model_keys)

    return algorithm_class(
        models,
        make_history_integrator=partial_make_history_integrator,
        compute_history_features=partial_compute_history_features,
    )

def make_a2c_algorithm_ais(
    name: str,
    env: Environment,
    *,
    truncated_histories: bool,
    truncated_histories_n: int,
) -> A2C_ABC_AIS:

    partial_make_history_integrator = functools.partial(
        make_history_integrator_AIS,
        truncated_histories=truncated_histories,
        truncated_histories_n=truncated_histories_n,
    )
    partial_compute_history_features = functools.partial(
        compute_history_features_AIS,
        truncated=truncated_histories,
        n=truncated_histories_n,
    )

    if name == 'a2c':
        algorithm_class = A2C_AIS
    elif name == 'asym-a2c':
        algorithm_class = AsymA2C_AIS
    # elif name == 'asym-a2c-state':
    #     algorithm_class = AsymA2C_State
    else:
        raise ValueError(f'invalid algorithm name {name}')

    models = make_models_AIS(env, keys=algorithm_class.model_keys)

    return algorithm_class(
        models,
        make_history_integrator_AIS=partial_make_history_integrator,
        compute_history_features_AIS=partial_compute_history_features,
    )


def make_dqn_algorithm(
    name: str,
    env: Environment,
    *,
    truncated_histories: bool,
    truncated_histories_n: int,
) -> DQN_ABC:

    partial_make_history_integrator = functools.partial(
        make_history_integrator,
        truncated_histories=truncated_histories,
        truncated_histories_n=truncated_histories_n,
    )
    partial_compute_history_features = functools.partial(
        compute_history_features,
        truncated=truncated_histories,
        n=truncated_histories_n,
    )

    if name == 'dqn':
        algorithm_class = DQN
    elif name == 'adqn':
        algorithm_class = ADQN
    elif name == 'adqn-bootstrap':
        algorithm_class = ADQN_Bootstrap
    elif name == 'adqn-state':
        algorithm_class = ADQN_State
    elif name == 'adqn-state-bootstrap':
        algorithm_class = ADQN_State_Bootstrap
    elif name == 'adqn-short':
        algorithm_class = ADQN_Short
    else:
        raise ValueError(f'invalid algorithm name {name}')

    models = make_models(env, keys=algorithm_class.model_keys)
    return algorithm_class(
        models,
        make_history_integrator=partial_make_history_integrator,
        compute_history_features=partial_compute_history_features,
    )
