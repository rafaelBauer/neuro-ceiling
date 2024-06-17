from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Final, Optional

import wandb

from policy import PolicyBase
from utils.device import device


@dataclass
class LearnAlgorithmConfig:
    _ALGO_TYPE: str = field(init=False, default="LearnAlgorithm")
    batch_size: int = field(init=True)
    learning_rate: float = field(init=True)
    weight_decay: float = field(init=True)
    dataset_path: str = field(init=False, default="")

    @property
    def algo_type(self) -> str:
        return self._ALGO_TYPE


class LearnAlgorithm:
    def __init__(self, config: LearnAlgorithmConfig, policy: PolicyBase):
        self._policy = policy
        self.__CONFIG: Final[LearnAlgorithmConfig] = config
        policy.to(device)
        wandb.watch(policy, log_freq=100)

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def train_step(self, num_steps: Optional[int] = None):
        pass
