from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy
import torch
from torch import Tensor

from utils.device import device
from utils.human_feedback import HumanFeedback
from utils.logging import logger
from utils.sceneobservation import SceneObservation


@dataclass(kw_only=True)
class FeedbackDeviceConfig:
    _DEVICE_TYPE: str = field(init=False)
    action_dim: int
    noise_distribution: list[list[float]] = field(default_factory=list)

    @property
    def device_type(self) -> str:
        return self._DEVICE_TYPE

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class FeedbackDevice:
    def __init__(self, config: FeedbackDeviceConfig, reset_feedback_after_evaluation: bool = True):
        self._CONFIG = config
        self._last_corrective_feedback: numpy.array = numpy.zeros(config.action_dim)
        self._reset_feedback_after_evaluation = reset_feedback_after_evaluation
        self.noise_distribution: numpy.ndarray = numpy.array(
            [numpy.divide(numpy.array(distribution), 100) for distribution in config.noise_distribution]
        )

    def sample_feedback(self, original_action: Tensor, scene_observation: SceneObservation) -> (Tensor, HumanFeedback):
        corrective_feedback = self.check_corrective_feedback(original_action, scene_observation)
        evaluative_feedback = self._compute_evaluative_feedback(original_action, corrective_feedback)
        returned_action = original_action
        if evaluative_feedback == HumanFeedback.CORRECTED:
            returned_action = corrective_feedback
        return returned_action, evaluative_feedback

    def check_corrective_feedback(self, original_action: Tensor, scene_observation: SceneObservation) -> Tensor:
        corrective_feedback = self._specific_corrective_feedback(scene_observation)
        corrective_feedback = corrective_feedback.to(device)
        corrective_feedback = self.add_noise_to_feedback(original_action, corrective_feedback)
        if self._reset_feedback_after_evaluation:
            self._last_corrective_feedback = numpy.zeros(self._last_corrective_feedback.size)
        return corrective_feedback

    def add_noise_to_feedback(self, original_action: Tensor, corrective_feedback: Tensor) -> Tensor:
        if (
            not torch.any(corrective_feedback)
            or self.noise_distribution.size == 0
            or torch.equal(original_action, corrective_feedback)
        ):
            return corrective_feedback

        label_noise_distribution = self.noise_distribution[corrective_feedback.nonzero()[0][0]]
        label_options = numpy.arange(self._CONFIG.action_dim)
        noisy_choice = numpy.random.choice(label_options, p=label_noise_distribution)
        noisy_feedback = numpy.zeros(self._CONFIG.action_dim)
        noisy_feedback[noisy_choice] = 1
        noisy_feedback_tensor = torch.tensor(noisy_feedback).to(device)
        if not torch.equal(noisy_feedback_tensor, corrective_feedback):
            logger.debug(f"Added noise to original feedback!: {corrective_feedback} -> {noisy_feedback}")
        return noisy_feedback_tensor

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
