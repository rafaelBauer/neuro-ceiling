from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Final, override

import numpy
from torch import Tensor, cat

from goal.goal import Goal
from policy.policy import PolicyBase, PolicyBaseConfig
from utils.keyboard_observer import KeyboardObserver
from utils.logging import log_constructor


@dataclass(kw_only=True)
class ManualPolicyConfig(PolicyBaseConfig):
    """
    Configuration class for ManualPolicy. Inherits from PolicyBaseConfig.
    """

    _POLICY_TYPE: str = field(init=False, default="ManualPolicy")


class ManualPolicy(PolicyBase):
    """
    ManualPolicy class that inherits from PolicyBase. This class is used to manually control the policy.
    """

    @log_constructor
    def __init__(self, config: ManualPolicyConfig, keyboard_observer: KeyboardObserver, **kwargs):
        """
        Constructor for the ManualPolicy class.

        This constructor initializes the ManualPolicy object with the provided configuration and keyword arguments.

        :param config: Configuration object for the ManualPolicy. This should be an instance of ManualPolicyConfig.
        :param kwargs: Additional keyword arguments.
        """
        super().__init__(config, **kwargs)
        self._keyboard_observer: Final[KeyboardObserver] = keyboard_observer
        self._CONFIG: ManualPolicyConfig = config

    @override
    def forward(self, states: Tensor) -> Tensor:
        # For when the keyboard observer is not working
        # action = numpy.array([0.0, 0.0, 0.0, -0.9, 0.0, 0.9])
        # gripper = numpy.array([0.0])
        action = self._keyboard_observer.get_ee_action()
        gripper = self._keyboard_observer.gripper
        return self.specific_forward(numpy.concatenate((action, gripper)))

    @override
    def update(self):
        """
        Update method for the ManualPolicy class. This method is currently not implemented.
        """

    @override
    def task_to_be_executed(self, goal: Goal):
        """
        Method to be executed when a task is to be executed. This method is currently not implemented.

        Parameters:
            goal: Goal object representing the task to be executed.
        """

    @abstractmethod
    def specific_forward(self, action: numpy.array):
        """
        Abstract method for the specific forward pass of the policy.

        Parameters:
            action (numpy.array): representing the action to be taken.

        Returns:
            Tensor: representing the output of the forward pass.
        """
