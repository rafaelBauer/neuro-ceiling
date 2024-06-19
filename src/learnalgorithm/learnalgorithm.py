from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Final, Optional

import wandb
from torch import Tensor, nn

from controller.controllerstep import ControllerStep
from goal.goal import Goal
from policy import PolicyBase
from utils.device import device
from utils.sceneobservation import SceneObservation


@dataclass
class LearnAlgorithmConfig:
    _ALGO_TYPE: str = field(init=False, default="LearnAlgorithm")
    batch_size: int = field(init=True)
    learning_rate: float = field(init=True)
    weight_decay: float = field(init=True)
    episode_steps: int = field(init=True)
    load_dataset: str = field(init=True, default="")
    save_dataset: str = field(init=True, default="")

    @property
    def algo_type(self) -> str:
        return self._ALGO_TYPE


class LearnAlgorithm:
    def __init__(self, config: LearnAlgorithmConfig, policy: PolicyBase):
        self._policy = policy
        self._CONFIG: Final[LearnAlgorithmConfig] = config
        policy.to(device)
        wandb.watch(policy, log_freq=100)

    @abstractmethod
    def train(self, mode: bool = True):
        pass

    @abstractmethod
    def step(self, controller_step: ControllerStep):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def load_from_file(self):
        pass
