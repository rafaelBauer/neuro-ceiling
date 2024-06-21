from dataclasses import dataclass, field
from typing import Optional

import numpy
import numpy as np
import torch
from overrides import override
from torch import Tensor

from goal.goal import Goal
from goal.pickplaceobject import PickPlaceObject
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

    It is expected that the user selects the object, and to which spot they want the object to be moved to.

    Attributes:
        __last_action (Goal): The last goal that was set.
    """

    def __init__(self, config: ManualObjectActionPolicyConfig, keyboard_observer: KeyboardObserver, **kwargs):
        super().__init__(config, keyboard_observer, **kwargs)
        self._keyboard_observer.subscribe_callback_to_direction(self.__key_pressed_callback)
        self.__last_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    @override
    def specific_forward(self, action: numpy.array, current_observation: SceneObservation) -> Tensor:
        label = torch.zeros(4)

        if self.__last_action[5] < -0.5:  # "u" key
            label[0] = True
        elif self.__last_action[4] < -0.5:  # "i" key
            label[1] = True
        elif self.__last_action[5] > 0.5:  # "o" key
            label[2] = True
        else:
            label[3] = True

        return label.unsqueeze(0)

    @override
    def episode_finished(self):
        self.__last_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

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
        self.__last_action = action
