from dataclasses import dataclass, field
from typing import Optional, TypeVar

import torch
from overrides import override

from controller import ControllerConfig, ControllerBase
from envs import BaseEnvironment
from learnalgorithm import LearnAlgorithm
from policy import PolicyBase
from utils.timer import Timer
from utils.logging import log_constructor


@dataclass
class PeriodicControllerConfig(ControllerConfig):
    _CONTROLLER_TYPE: str = field(init=False, default="PeriodicController")
    polling_period_s: float = 0.05


ActionType = TypeVar("ActionType")


class PeriodicController(ControllerBase[ActionType]):
    @log_constructor
    def __init__(
        self,
        config: PeriodicControllerConfig,
        environment: BaseEnvironment,
        policy: PolicyBase,
        child_controller: Optional[ControllerBase] = None,
        learn_algorithm: Optional[LearnAlgorithm] = None,
    ):
        self.__timer = Timer(self._timer_callback, config.polling_period_s)
        super().__init__(config, environment, policy, child_controller, learn_algorithm)

    @override
    def _specific_start(self):
        self.__timer.start()

    @override
    def _specific_stop(self):
        self.__timer.stop()

    def _timer_callback(self):
        # Lock it so the previous observation is not changed while we are using it
        with self._control_variables_lock:
            next_action = self._policy(self._previous_observation)
            if isinstance(next_action, torch.Tensor):
                next_action = next_action.to("cpu")
                next_action = self._action_type.from_tensor(next_action.squeeze(0).detach())

        if next_action is not None:
            self._step(next_action)
