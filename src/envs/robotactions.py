import typing
from abc import abstractmethod
from enum import IntEnum

import numpy
import torch

from utils.pose import Pose


class GripperCommand(IntEnum):
    """
    An enumeration representing the commands that can be sent to the gripper.

    Attributes
    ----------
    OPEN : int
       A command to open the gripper. The value of this command is 1.
    CLOSE : int
       A command to close the gripper. The value of this command is -1.
    """

    OPEN = 1
    CLOSE = -1


class RobotAction(torch.Tensor):
    def __new__(cls, x, gripper_command, *args, **kwargs):
        return super().__new__(cls, x, *args, **kwargs)

    def __init__(self, gripper_command: GripperCommand, *args, **kwargs):
        self._gripper_command: GripperCommand = gripper_command
        super().__init__(*args, **kwargs)

    @property
    def gripper_command(self) -> GripperCommand:
        return self._gripper_command

    @abstractmethod
    def get_raw_action(self) -> numpy.ndarray:
        pass


class DeltaEEPoseAction(RobotAction):
    def __init__(self, delta_pose: Pose, gripper_command: GripperCommand, *args, **kwargs):
        self.__delta_pose = delta_pose
        super().__init__(gripper_command, *args, **kwargs)

    @property
    def delta_pose(self):
        return self.__delta_pose

    @typing.override
    def get_raw_action(self) -> numpy.ndarray:
        pass


class TargetJointPositionAction(RobotAction):
    def __init__(self, target_position: numpy.ndarray, gripper_command: GripperCommand, *args, **kwargs):
        self.__target_position: numpy.ndarray = target_position
        super().__init__(gripper_command, *args, **kwargs)

    @property
    def target_position(self):
        return self.__target_position

    @typing.override
    def get_raw_action(self) -> numpy.ndarray:
        return numpy.hstack([self.__target_position, self.gripper_command])
