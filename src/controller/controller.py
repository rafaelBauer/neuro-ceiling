import threading
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Final, Any, Optional

import numpy as np

from envs import BaseEnvironment
from learnalgorithm.learnalgorithm import LearnAlgorithmBaseConfig
from policy import PolicyBase
from goal.goal import Goal
from utils.logging import log_constructor, logger


@dataclass
class ControllerConfig:
    _CONTROLLER_TYPE: str = field(init=True)
    learn_algorithm_config: LearnAlgorithmBaseConfig

    @property
    def controller_type(self) -> str:
        return self._CONTROLLER_TYPE


class ControllerBase:
    @log_constructor
    def __init__(
        self,
        config: ControllerConfig,
        environment: BaseEnvironment,
        policy: PolicyBase,
        child_controller: Optional['ControllerBase'] = None,
    ):
        self.__CONFIG: Final[ControllerConfig] = config
        self._environment: Final[BaseEnvironment] = environment
        self._policy: Final[PolicyBase] = policy

        self._child_controller: Optional[ControllerBase] = child_controller

        # Control variables for learning
        self._current_observation: np.array = np.array([])
        self._current_reward: float = 0.0
        self._episode_finished: bool = False
        self._current_info: dict[str, Any] = {}
        self._control_variables_lock: threading.Lock = threading.Lock()

        self._goal: Goal = Goal()

        if self._child_controller is not None:
            self._action_step_function = self._child_controller.set_goal
        else:
            self._action_step_function = self._environment.step

    def start(self):
        if self._child_controller is not None:
            self._child_controller.start()
        self._specific_start()

    @abstractmethod
    def _specific_start(self):
        pass

    def stop(self):
        if self._child_controller is not None:
            self._child_controller.stop()
        self._specific_stop()

    @abstractmethod
    def _specific_stop(self):
        pass

    def set_goal(self, goal: Goal):
        if not self._goal == goal:
            self._goal = goal
            logger.debug("Setting goal to policy: {}", self._goal)
            self._policy.task_to_be_executed(self._goal)
        with self._control_variables_lock:
            return self._current_observation, self._current_reward, self._episode_finished, self._current_info
