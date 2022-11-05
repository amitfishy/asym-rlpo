import torch.nn as nn

from .base import Representation


class RNNHistoryRepresentationAIS(Representation):
    def __init__(
        self,
        action_model: Representation,
        observation_model: Representation,
        reward_model: Representation,
        *,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = 'relu',
    ):
        super().__init__()
        self.action_model = action_model
        self.observation_model = observation_model
        self.reward_model = reward_model
        input_size = self.action_model.dim + self.observation_model.dim + 1
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,
        )

    @property
    def dim(self):
        return self.rnn.hidden_size

    def forward(self, inputs, *, hidden=None):
        return self.rnn(inputs, hidden)


class GRUHistoryRepresentationAIS(Representation):
    def __init__(
        self,
        action_model: Representation,
        observation_model: Representation,
        reward_model: Representation,
        *,
        hidden_size: int,
        num_layers: int = 1,
    ):
        super().__init__()
        self.action_model = action_model
        self.observation_model = observation_model
        self.reward_model = reward_model
        input_size = self.action_model.dim + self.observation_model.dim + 1
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    @property
    def dim(self):
        return self.rnn.hidden_size

    def forward(self, inputs, *, hidden=None):
        return self.rnn(inputs, hidden)
