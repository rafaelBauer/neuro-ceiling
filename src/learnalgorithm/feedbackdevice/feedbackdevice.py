from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy
import torch
from torch import Tensor

from utils.device import device
from utils.human_feedback import HumanFeedback
from utils.sceneobservation import SceneObservation


@dataclass
class FeedbackDeviceConfig:
    _DEVICE_TYPE: str = field(init=False)
    action_dim: int = field(init=True)

    @property
    def device_type(self) -> str:
        return self._DEVICE_TYPE

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class FeedbackDevice:
    def __init__(self, config: FeedbackDeviceConfig, reset_feedback_after_evaluation: bool = True):
        self._last_corrective_feedback: numpy.array = numpy.zeros(config.action_dim)
        self._reset_feedback_after_evaluation = reset_feedback_after_evaluation
        pass

    def sample_feedback(self, original_action: Tensor, scene_observation: SceneObservation) -> (Tensor, HumanFeedback):
        corrective_feedback = self.check_corrective_feedback(scene_observation)
        evaluative_feedback = self._compute_evaluative_feedback(original_action, corrective_feedback)
        returned_action = original_action
        if evaluative_feedback == HumanFeedback.CORRECTED:
            returned_action = corrective_feedback
        return returned_action, evaluative_feedback

    def check_corrective_feedback(self, scene_observation: SceneObservation) -> Tensor:
        corrective_feedback = self._specific_corrective_feedback(scene_observation)
        corrective_feedback = corrective_feedback.to(device)
        if self._reset_feedback_after_evaluation:
            self._last_corrective_feedback = numpy.zeros(self._last_corrective_feedback.size)
        return corrective_feedback

    def _compute_evaluative_feedback(self, original_action: Tensor, corrected_action: Tensor) -> HumanFeedback:
        # If there was no input, it means there is no corrective feedback, so we return the current evaluative feedback.
        if not torch.any(corrected_action):
            return self.get_evaluative_feedback()
        elif torch.equal(original_action, corrected_action):
            return HumanFeedback.GOOD
        return HumanFeedback.CORRECTED

    @property
    def has_corrective_feedback(self) -> bool:
        return numpy.any(self._last_corrective_feedback)

    @abstractmethod
    def get_evaluative_feedback(self) -> HumanFeedback:
        pass

    def reset(self):
        self._last_corrective_feedback = numpy.zeros(self._last_corrective_feedback.size)

    @abstractmethod
    def _specific_reset(self):
        pass

    @abstractmethod
    def _specific_corrective_feedback(self, scene_observation: SceneObservation) -> Tensor:
        pass

    @abstractmethod
    def subscribe_callback_to_evaluative_feedback(self, callback: Callable[[HumanFeedback], None]):
        pass
