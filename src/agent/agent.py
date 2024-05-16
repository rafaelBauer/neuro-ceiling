from dataclasses import dataclass
import time
from typing import Final

import numpy as np

from envs import BaseEnvironment
from learnalgorithm.learnalgorithm import LearnAlgorithmBaseConfig
from policy import PolicyBase
from utils.logging import log_constructor
from utils.timer import Timer


@dataclass
class AgentConfig:
    polling_period_s: float
    learn_algorithm_config: LearnAlgorithmBaseConfig


class AgentBase:
    @log_constructor
    def __init__(self, config: AgentConfig, environment: BaseEnvironment, policy: PolicyBase):
        self.__CONFIG: Final[AgentConfig] = config
        self.__timer = Timer(self._timer_callback, self.__CONFIG.polling_period_s)
        self.__environment: Final[BaseEnvironment] = environment
        self.__policy: Final[PolicyBase] = policy

    def start(self):
        self.__timer.start()

    def stop(self):
        self.__timer.stop()

    def _timer_callback(self):
        state: np.array = np.zeros(7)
        action: np.array = self.__policy(state)
        self.__environment.step(action)
