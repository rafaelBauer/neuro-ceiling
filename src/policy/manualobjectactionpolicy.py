from dataclasses import dataclass, field
from typing import Optional

import numpy
import numpy as np
import torch
from overrides import override
from torch import Tensor

from goal.goal import Goal
from goal.pickplaceobject import PickPlaceObject
from learnalgorithm.feedbackdevice.feedbackdevice import FeedbackDevice
from policy.manualpolicy import ManualPolicyConfig, ManualPolicy
from utils.gripperstate import GripperState
from utils.keyboard_observer import KeyboardObserver
from utils.logging import logger
from utils.pose import Pose
from utils.sceneobservation import SceneObservation


@dataclass(kw_only=True)
class ManualObjectActionPolicyConfig(ManualPolicyConfig):
    """
    Configuration class for ManualObjectActionPolicyConfig. Inherits from ManualPolicy.
    """

    _POLICY_TYPE: str = field(init=False, default="ManualObjectActionPolicy")


class ManualObjectActionPolicy(ManualPolicy):
    """
    The ManualObjectActionPolicy class represents a policy for manually generate the MoveObjectToPosition objects.
    It inherits from the ManualPolicy class.

    It is expected that the user selects an object or spot in which the object should be picked or placed into.

    Attributes:
        __last_feedback (numpy.array): The last feedback that was set.
    """

    def __init__(self, config: ManualObjectActionPolicyConfig, keyboard_observer: KeyboardObserver, **kwargs):
        super().__init__(config, keyboard_observer, **kwargs)

    @override
    def specific_forward(self, action: numpy.array, current_observation: SceneObservation) -> Tensor:
        return torch.Tensor(action).unsqueeze(0)

    @override
    def episode_finished(self):
        self._feedback_device.reset()
