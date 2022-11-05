import abc
import copy
from typing import ClassVar, Dict, List, Union

import torch
import torch.nn as nn

from asym_rlpo.features_ais import HistoryFeaturesComputerAIS, HistoryIntegratorMakerAIS

ModelKeysList = List[str]
ModelKeysDict = Dict[str, ModelKeysList]
ModelKeysDict = Dict[str, Union[ModelKeysDict, ModelKeysList]]


class Algorithm_ABC_AIS(metaclass=abc.ABCMeta):
    model_keys: ClassVar[ModelKeysDict]

    def __init__(
        self,
        models: nn.ModuleDict,
        *,
        make_history_integrator_AIS: HistoryIntegratorMakerAIS,
        compute_history_features_AIS: HistoryFeaturesComputerAIS,
    ):
        super().__init__()
        self.models = models
        self.target_models = copy.deepcopy(models)

        self.make_history_integrator_AIS = make_history_integrator_AIS
        self.compute_history_features_AIS = compute_history_features_AIS

    def to(self, device: torch.device):
        self.models.to(device)
        self.target_models.to(device)
