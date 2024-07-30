from dataclasses import dataclass, field
from typing import Callable

import numpy
import torch
from overrides import override
from torch import Tensor

from learnalgorithm.feedbackdevice.feedbackdevice import FeedbackDevice, FeedbackDeviceConfig
from utils.human_feedback import HumanFeedback
from utils.keyboard_observer import KeyboardObserver
from utils.sceneobservation import SceneObservation


@dataclass
class KeyboardFeedbackConfig(FeedbackDeviceConfig):
    _DEVICE_TYPE: str = field(init=False, default="KeyboardFeedback")

    @property
    @override
    def name(self) -> str:
        return "keyboard"


class KeyboardFeedback(FeedbackDevice):
    def __init__(
        self,
        config: KeyboardFeedbackConfig,
        keyboard_observer: KeyboardObserver,
        reset_feedback_after_evaluation: bool = True,
    ):
        self._keyboard_observer: KeyboardObserver = keyboard_observer
        self._keyboard_observer.subscribe_callback_to_direction(self.__key_pressed_callback)
        super().__init__(config, reset_feedback_after_evaluation)

    @override
    def subscribe_callback_to_evaluative_feedback(self, callback: Callable[[HumanFeedback], None]):
        self._keyboard_observer.subscribe_callback_to_label(callback)

    def get_evaluative_feedback(self) -> HumanFeedback:
        return self._keyboard_observer.label

    @override
    def _specific_reset(self):
        self._keyboard_observer.reset()

    @override
    def _specific_corrective_feedback(self, scene_observation: SceneObservation) -> Tensor:
        return torch.tensor(self._last_corrective_feedback)

    def __key_pressed_callback(self, action: numpy.array):
        """
        This function is a callback that is triggered when a key is pressed.

        It checks if any action has been performed. If not, it returns and does nothing.
        If an action has been performed, it updates the last action and sets the new command flag to True.

        Args:
            action (numpy.array): An array representing the action performed.
        """
        if not numpy.any(action):
            return

        if action[5] > 0.5:  # "o" key
            self._last_corrective_feedback[0] = True
        elif action[4] < -0.5:  # "i" key
            self._last_corrective_feedback[1] = True
        elif action[5] < -0.5:  # "u" key
            self._last_corrective_feedback[2] = True
        else:
            self._last_corrective_feedback[3] = True
