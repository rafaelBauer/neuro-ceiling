from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Final

import numpy
from overrides import override
from torch import Tensor

from goal.goal import Goal
from learnalgorithm import create_feedback_device
from learnalgorithm.feedbackdevice.feedbackdevice import FeedbackDevice, FeedbackDeviceConfig
from policy.policy import PolicyBase, PolicyBaseConfig
from utils.keyboard_observer import KeyboardObserver
from utils.logging import log_constructor
from utils.sceneobservation import SceneObservation


@dataclass(kw_only=True)
class ManualPolicyConfig(PolicyBaseConfig):
    """
    Configuration class for ManualPolicy. Inherits from PolicyBaseConfig.
    """

    _POLICY_TYPE: str = field(init=False, default="ManualPolicy")
    feedback_device_config: FeedbackDeviceConfig = field(init=True)


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
        self._feedback_device: Final[FeedbackDevice] = create_feedback_device(
            config.feedback_device_config, keyboard_observer=keyboard_observer
        )
        self._CONFIG: ManualPolicyConfig = config

    @override
    def forward(self, states) -> Tensor:
        # For when the keyboard observer is not working
        # action = numpy.array([0.0, 0.0, 0.0, -0.9, 0.0, 0.9, 0.0])
        assert isinstance(states, SceneObservation), "states should be of type SceneObservation"
        action = self._feedback_device.check_corrective_feedback(states).numpy()
        return self.specific_forward(action, states)

    @override
    def goal_to_be_achieved(self, goal: Goal):
        """
        Method to be executed when a task is to be executed. This method is currently not implemented.

        Parameters:
            goal: Goal object representing the task to be executed.
        """

    @abstractmethod
    def specific_forward(self, action: numpy.array, current_observation: SceneObservation) -> Tensor:
        """
        Abstract method for the specific forward pass of the policy.

        Parameters:
            action (numpy.array): representing the action to be taken.
            current_observation (SceneObservation): representing the current observation.

        Returns:
            Tensor: representing the output of the forward pass.
        """
