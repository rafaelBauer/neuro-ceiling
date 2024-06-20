from dataclasses import dataclass, field
from typing import Optional

import numpy
import numpy as np
import torch
from overrides import override
from torch import Tensor

from envs.scene import Scene
from goal.goal import Goal
from goal.pickplaceobject import PickPlaceObject
from policy.manualpolicy import ManualPolicyConfig, ManualPolicy
from utils.gripperstate import GripperState
from utils.keyboard_observer import KeyboardObserver
from utils.labeltoobjectpose import LabelToGoalTranslator
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
    __CUBE_SIZE: float = 0.04
    __CUBE_HALF_SIZE: float = 0.02
    """
    The ManualObjectActionPolicy class represents a policy for manually generate the MoveObjectToPosition objects.
    It inherits from the ManualPolicy class.

    It is expected that the user selects the object, and to which spot they want the object to be moved to.

    Attributes:
        __last_goal (Goal): The last goal that was set.
    """

    def __init__(self, config: ManualObjectActionPolicyConfig, keyboard_observer: KeyboardObserver, **kwargs):
        self.__last_goal: Goal = Goal()
        super().__init__(config, keyboard_observer, **kwargs)
        self._keyboard_observer.subscribe_callback_to_direction(self.__key_pressed_callback)
        self.__last_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.__new_command = False
        self.__last_gripper_state: GripperState = GripperState.OPENED
        self.__label_to_goal_translator = LabelToGoalTranslator()

    @override
    def specific_forward(self, action: numpy.array, current_observation: SceneObservation) -> Tensor:
        target_pose: Optional[Pose] = None

        # If statement only to protect against very first iteration where the current_observation is empty
        if len(current_observation.spots.values()) == 0:
            return self.__last_goal

        logger.debug("Current gripper state is {}", current_observation.gripper_state.name)

        # If statement not placed with previous one on purpose. I want to see the gripper state even if there are no
        # actions to be taken
        if not self.__new_command:
            return self.__last_goal

        label = torch.zeros(3)

        if self.__last_action[5] < -0.5:  # "u" key
            label[0] = True
        elif self.__last_action[4] < -0.5:  # "i" key
            label[1] = True
        elif self.__last_action[5] > 0.5:  # "o" key
            label[2] = True

        new_goal = self.__label_to_goal_translator.translate_label_to_pickplaceobject(label, current_observation)

        self.__new_command = False

        if new_goal != self.__last_goal:
            self.__last_goal = new_goal

        return self.__last_goal

    @override
    def episode_finished(self):
        self.__last_goal = Goal()

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
        self.__new_command = True
