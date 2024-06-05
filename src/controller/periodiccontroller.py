from dataclasses import dataclass, field
from typing import override, Final, Optional

from controller import ControllerConfig, ControllerBase
from envs import BaseEnvironment
from policy import PolicyBase
from utils.timer import Timer
from utils.logging import log_constructor


@dataclass
class PeriodicControllerConfig(ControllerConfig):
    _CONTROLLER_TYPE: str = field(init=False, default="PeriodicController")
    polling_period_s: float = 0.05


class PeriodicController(ControllerBase):
    @log_constructor
    def __init__(
        self,
        config: PeriodicControllerConfig,
        environment: BaseEnvironment,
        policy: PolicyBase,
        child_controller: Optional[ControllerBase] = None,
    ):
        self.__timer = Timer(self._timer_callback, config.polling_period_s)
        super().__init__(config, environment, policy, child_controller)

    @override
    def _specific_start(self):
        self.__timer.start()

    @override
    def _specific_stop(self):
        self.__timer.stop()

    def _timer_callback(self):
        action = self._policy(self._current_observation)

        if action is not None:
            with self._control_variables_lock:
                (self._current_observation, self._current_reward, self._episode_finished, self._current_info) = (
                    self._action_step_function(action)
                )