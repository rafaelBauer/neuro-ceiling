from dataclasses import dataclass, field
from enum import Enum
from typing import Final, override

import numpy
import torch
import transforms3d
from torch import Tensor, cat

from envs.robotactions import DeltaEEPoseAction, GripperCommand
from goal.goal import Goal
from goal.movetoposition import MoveObjectToPosition
from policy.policy import PolicyBase, PolicyBaseConfig
from utils.keyboard_observer import KeyboardObserver
from utils.logging import log_constructor
from utils.pose import Pose, RotationRepresentation


class PolicyLevel(Enum):
    LOW_LEVEL = "low_level",
    HIGH_LEVEL = "high_level"


@dataclass(kw_only=True)
class ManualPolicyConfig(PolicyBaseConfig):
    """
    Configuration class for ManualPolicy. Inherits from PolicyBaseConfig.
    """

    _POLICY_TYPE: str = field(init=False, default="ManualPolicy")
    _POLICY_LEVEL: PolicyLevel = field(init=True)

    @property
    def policy_level(self) -> PolicyLevel:
        return self._POLICY_LEVEL


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
        self.__keyboard_observer: Final[KeyboardObserver] = keyboard_observer
        self._CONFIG: ManualPolicyConfig = config

    @override
    def forward(self, states: Tensor) -> Tensor:
        # For when the keyboard observer is not working
        action = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.9])
        gripper = numpy.array([0.0])
        # action = self.__keyboard_observer.get_ee_action()
        # gripper = self.__keyboard_observer.gripper

        return_val = None
        if self._CONFIG.policy_level == PolicyLevel.HIGH_LEVEL:
            object_pose = Pose(p=[0.615, 0, 0.02], q=[0, 1, 0, 0])
            target_pose = Pose(p=[0.615, 0.2, 0.06], q=[0, 1, 0, 0])
            return_val = MoveObjectToPosition(object_pose, target_pose)
        else:
            if gripper < 0:
                gripper_command = GripperCommand.CLOSE
            else:
                gripper_command = GripperCommand.OPEN
            delta_pose = Pose(p=action[:3], euler=action[3:])
            return_val = DeltaEEPoseAction(delta_pose, rotation_representation=RotationRepresentation.EULER,
                                           gripper_command=gripper_command)
        return return_val

    @override
    def update(self):
        """
        Update method for the ManualPolicy class. This method is currently not implemented.
        """
        pass

    @override
    def task_to_be_executed(self, task: Goal):
        pass
