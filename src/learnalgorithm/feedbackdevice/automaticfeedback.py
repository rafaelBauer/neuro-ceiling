import math
from dataclasses import dataclass, field
from typing import Callable

import numpy
import torch
from overrides import override
from torch import Tensor

from envs.taskconfig import TaskConfig
from learnalgorithm.feedbackdevice.feedbackdevice import FeedbackDevice, FeedbackDeviceConfig
from utils.human_feedback import HumanFeedback
from utils.labeltoobjectpose import LabelToPoseTranslator
from utils.logging import logger
from utils.pose import Pose
from utils.sceneobservation import SceneObservation


@dataclass(kw_only=True)
class AutomaticFeedbackConfig(FeedbackDeviceConfig):
    _DEVICE_TYPE: str = field(init=False, default="AutomaticFeedback")
    task_config: TaskConfig = field(init=True)
    corrective_probability: int = field(
        init=True, default=100
    )  # Ratio of corrective feedback over evaluative feedback. 100 means 100% corrective feedback and 0% evaluative feedback

    @property
    @override
    def name(self) -> str:
        return (
            "automatic_"
            + str(self.corrective_probability)
            + "_corr_"
            + str(100 - self.corrective_probability)
            + "_eval"
        )

    @property
    def corrective_probability_bounded(self) -> float:
        return self.corrective_probability / 100


class AutomaticFeedback(FeedbackDevice):
    def __init__(self, config: AutomaticFeedbackConfig, keyboard_observer=None):
        self._CONFIG = config
        super().__init__(config)

    @override
    def subscribe_callback_to_evaluative_feedback(self, callback: Callable[[HumanFeedback], None]):
        pass

    def get_evaluative_feedback(self) -> HumanFeedback:
        return HumanFeedback.GOOD

    def _compute_evaluative_feedback(self, original_action: Tensor, corrected_action: Tensor) -> HumanFeedback:
        temporary_feedback = super()._compute_evaluative_feedback(original_action, corrected_action)
        if temporary_feedback == HumanFeedback.CORRECTED:
            temporary_feedback = self.__sample_evaluative_or_corrective_feedback()
            logger.debug(f"Automatic feedback chose feedback: {temporary_feedback.name}")
        return temporary_feedback

    def __sample_evaluative_or_corrective_feedback(self):
        sample_number = numpy.random.uniform()
        corrective_prob = self._CONFIG.corrective_probability_bounded
        if corrective_prob >= sample_number:
            # If it is corrected, then the corrective feedback will be chosen, otherwise it will be evaluative.
            return HumanFeedback.CORRECTED
        return HumanFeedback.BAD

    @override
    def _specific_reset(self):
        pass

    @override
    def _specific_corrective_feedback(self, scene_observation: SceneObservation) -> Tensor:
        _, is_object_being_held = LabelToPoseTranslator.is_object_being_held_by_end_effector(scene_observation)

        sorted_by_z = sorted(self._CONFIG.task_config.target_objects_pose.items(), key=lambda x: x[1].p[2])
        if math.isnan(sorted_by_z[0][1].p[1]):
            target_y = scene_observation.objects[sorted_by_z[0][0]][1]
        else:
            target_y = sorted_by_z[0][1].p[1]

        for i, (name, target_pose) in enumerate(sorted_by_z):
            object_pose = Pose(
                p=scene_observation.objects[name][:3].numpy(), euler=scene_observation.objects[name][3:].numpy()
            )
            target_pose.p = [target_pose.p[0], target_y, target_pose.p[2]]
            if not target_pose.is_close(object_pose, atol=0.01):
                # If object not being held, then get label of object pose.
                if not is_object_being_held:
                    self._last_feedback = self.__get_label_from_pose(object_pose)
                else:
                    self._last_feedback = self.__get_label_from_pose(target_pose)
                break
        return torch.Tensor(self._last_feedback)

    def __get_label_from_pose(self, pose: Pose) -> numpy.ndarray:
        # Hack for now
        if -0.15 >= pose.p[1]:
            return numpy.array([1, 0, 0, 0])
        elif -0.15 < pose.p[1] < 0.15:
            return numpy.array([0, 1, 0, 0])
        elif 0.15 <= pose.p[1]:
            return numpy.array([0, 0, 1, 0])
        else:
            return numpy.array([0, 0, 0, 1])
