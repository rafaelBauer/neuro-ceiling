from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Final

import numpy
import torch
from overrides import override
from torch import nn, Tensor

from goal.goal import Goal
from policy.policy import PolicyBase, PolicyBaseConfig
from utils.keyboard_observer import KeyboardObserver
from utils.logging import log_constructor
from utils.sceneobservation import SceneObservation


@dataclass(kw_only=True)
class CeilingPolicyConfig(PolicyBaseConfig):
    _POLICY_TYPE: str = field(init=False, default="CeilingPolicy")
    visual_embedding_dim: int = field(init=True)
    proprioceptive_dim: int = field(init=True)
    action_dim: int = field(init=True)


class CeilingPolicy(PolicyBase):
    @log_constructor
    def __init__(self, config: CeilingPolicyConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.__visual_encoding_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(start_dim=1),
        )
        lstm_dim = config.visual_embedding_dim + config.proprioceptive_dim
        self.__action_net = nn.Sequential(
            nn.LSTM(lstm_dim, lstm_dim), nn.Linear(lstm_dim, config.action_dim), nn.Tanh()
        )

        nn.GaussianNLLLoss
        self._CONFIG: CeilingPolicyConfig = config

    @override
    def forward(self, states: SceneObservation) -> Tensor:
        assert isinstance(states, SceneObservation), "states should be of type SceneObservation"
        visual_embedding = self.__visual_encoding_net(states.camera_observation)
        low_dim_input = torch.cat((visual_embedding, states.proprioceptive_obs), dim=1).unsqueeze(0)
        out = self.__action_net(low_dim_input)
        return out

    @override
    def goal_to_be_achieved(self, goal: Goal):
        """
        Method to be executed when a task is to be executed. This method is currently not implemented.

        Parameters:
            goal: Goal object representing the task to be executed.
        """
