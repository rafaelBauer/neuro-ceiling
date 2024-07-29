from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy
import torch
from torch import Tensor

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
        self._last_feedback: numpy.array = numpy.zeros(config.action_dim)
        self._reset_feedback_after_evaluation = reset_feedback_after_evaluation
        pass

    def check_corrective_feedback(self, scene_observation: SceneObservation) -> Tensor:
        corrective_feedback = self._specific_corrective_feedback(scene_observation)
        if self._reset_feedback_after_evaluation:
            self._last_feedback = numpy.zeros(self._last_feedback.size)
        return corrective_feedback

    @property
    def has_corrective_feedback(self) -> bool:
        return numpy.any(self._last_feedback)

    @abstractmethod
    def get_evaluative_feedback(self) -> HumanFeedback:
        pass

    def reset(self):
        self._last_feedback = numpy.zeros(self._last_feedback.size)

    @abstractmethod
    def _specific_reset(self):
        pass

    @abstractmethod
    def _specific_corrective_feedback(self, scene_observation: SceneObservation) -> Tensor:
        pass

    @abstractmethod
    def subscribe_callback_to_evaluative_feedback(self, callback: Callable[[HumanFeedback], None]):
        pass
