from dataclasses import dataclass
import time
from typing import Final

import numpy as np

from envs import BaseEnvironmentConfig, create_environment
from learnalgorithm.learnalgorithm import LearnAlgorithmBaseConfig
from policy import PolicyBaseConfig, create_policy
from utils.timer import Timer


@dataclass
class AgentConfig:
    polling_period_s: float
    learn_algorithm_config: LearnAlgorithmBaseConfig
    policy_config: PolicyBaseConfig
    environment: BaseEnvironmentConfig


class AgentBase:
    def __init__(self, config: AgentConfig):
        self.__CONFIG: Final[AgentConfig] = config
        self.__timer = Timer(self._timer_callback, self.__CONFIG.polling_period_s)
        self.environment = create_environment(self.__CONFIG.environment)
        self.__policy = create_policy(self.__CONFIG.policy_config)

    def start(self):
        self.environment.start()
        time.sleep(5)
        self.__timer.start()

    def stop(self):
        self.__timer.stop()

    def _timer_callback(self):
        state: np.array = np.zeros(7)
        # self.environment
        action: np.array = self.__policy(state)
        self.environment.step(action)
