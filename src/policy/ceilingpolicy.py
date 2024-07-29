from dataclasses import dataclass, field

import numpy
import torch
from overrides import override
from torch import nn, Tensor

from goal.goal import Goal
from policy.policy import PolicyBase, PolicyBaseConfig
from utils.device import device
from utils.logging import log_constructor
from utils.sceneobservation import SceneObservation


@dataclass(kw_only=True)
class CeilingPolicyConfig(PolicyBaseConfig):
    _POLICY_TYPE: str = field(init=False, default="CeilingPolicy")
    from_file: str = field(init=True, default="")
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
        self.__visual_encoding_net = self.__visual_encoding_net.to(device)

        LSTM_DIM = config.visual_embedding_dim + config.proprioceptive_dim
        # Created LSTM separated, because it stores the LSTM state between calls to forward.
        # At every new episode, this should be reset to None by calling episode_finished
        self.__lstm = nn.LSTM(LSTM_DIM, LSTM_DIM)
        self.__lstm_state: tuple[torch.Tensor, torch.Tensor] | None = None

        self.__action_net = nn.Sequential(nn.Linear(LSTM_DIM, config.action_dim), nn.Tanh(), nn.Softmax(dim=-1))
        self.__action_net = self.__action_net.to(device)

        self._CONFIG: CeilingPolicyConfig = config

    @override
    def forward(self, states) -> Tensor:
        # Expects a tensor of shape (batch_size, feature_size)
        if isinstance(states, list):
            scene_observation = states[0]
            lstm_state = states[1]
        else:
            scene_observation = states
            lstm_state = self.__lstm_state
        assert isinstance(scene_observation, SceneObservation), "states should be of type SceneObservation"

        if len(scene_observation.proprioceptive_obs) == 0:
            return torch.zeros(1, self._CONFIG.action_dim)
        scene_observation = scene_observation.to(device)
        input_tensor = scene_observation.camera_observation["rgb"].float()
        input_tensor = input_tensor.to(device)
        visual_embedding = self.__visual_encoding_net(input_tensor)
        low_dim_input = torch.hstack((visual_embedding, scene_observation.proprioceptive_obs)).unsqueeze(0)

        lstm_out, lstm_state = self.__lstm(low_dim_input, lstm_state)
        out = self.__action_net(lstm_out)

        # Want to write the lstm_state back to the input variable. If it is a list, we write it back to it
        if isinstance(states, list):
            states[1] = lstm_state
        else:
            self.__lstm_state = lstm_state
        # The output is a tensor of shape (time_step, batch_size, action_dim), and we always compute one time step,
        # therefore, we simply remove the first dimension
        return out.squeeze(0)

    @override
    def episode_finished(self):
        self.__lstm_state = None

    @override
    def goal_to_be_achieved(self, goal: Goal):
        """
        Method to be executed when a task is to be executed. This method is currently not implemented.

        Parameters:
            goal: Goal object representing the task to be executed.
        """
