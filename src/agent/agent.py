import threading
from dataclasses import dataclass
import time
from typing import Final, Any

import numpy as np

from envs import BaseEnvironment
from learnalgorithm.learnalgorithm import LearnAlgorithmBaseConfig
from policy import PolicyBase
from task.task import Task
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

        # Control variables for learning
        self.__current_observation: np.array = np.array([])
        self.__current_reward: float = 0.0
        self.__episode_finished: bool = False
        self.__current_info: dict[str, Any] = {}
        self.__policy_lock: threading.Lock = threading.Lock()

    def start(self):
        self.__timer.start()

    def stop(self):
        self.__timer.stop()
        self.__environment.stop()

    def execute_task(self, task: Task):
        with self.__policy_lock:
            self.__policy.plan_task(task)

    def _timer_callback(self):
        action: np.array
        with self.__policy_lock:
            action = self.__policy(self.__current_observation)

        if action is not None:
            (self.__current_observation,
             self.__current_reward,
             self.__episode_finished,
             self.__current_info) = self.__environment.step(action)
