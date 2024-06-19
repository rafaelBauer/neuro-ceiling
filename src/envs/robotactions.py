import typing
from abc import abstractmethod
from enum import IntEnum

import numpy
import torch
from mplib.pymp.kinematics import pinocchio
from overrides import override
from tensordict.prototype import tensorclass
from torch import Tensor

from utils.pose import Pose, RotationRepresentation


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

    @classmethod
    def from_tensor(cls, input_tensor: torch.Tensor) -> "GripperCommand":
        if 0.0 <= input_tensor:
            return cls.OPEN
        elif 0.0 > input_tensor:
            return cls.CLOSE


@tensorclass
class RobotAction:
    _gripper_command: Tensor

    def __init__(self, gripper_command: GripperCommand):
        self._gripper_command: Tensor = torch.tensor(gripper_command)
        # super().__init__(*args, **kwargs)

    @property
    def gripper_command(self) -> GripperCommand:
        return GripperCommand(self._gripper_command.item())

    @abstractmethod
    def to_tensor(self) -> torch.Tensor:
        pass

    @abstractmethod
    def to_delta_ee_pose(
        self, pinocchio_model: pinocchio.PinocchioModel, ee_link_index: int, current_pose: Pose
    ) -> "RobotAction":
        pass

    @abstractmethod
    def to_target_joint_position(self) -> "RobotAction":
        pass


class PoseActionBase(RobotAction):
    def __init__(
        self,
        pose: Pose,
        rotation_representation: RotationRepresentation,
        gripper_command: GripperCommand,
    ):
        self.__pose: Pose = pose
        self.__rotation_representation: RotationRepresentation = rotation_representation
        super().__init__(gripper_command=gripper_command)

    @property
    def pose(self) -> Pose:
        return self.__pose

    @property
    def rotation_representation(self):
        return self.__rotation_representation

    @override
    def to_tensor(self) -> torch.Tensor:
        return torch.hstack([self.pose.to_tensor(self.rotation_representation), self._gripper_command])


class DeltaEEPoseAction(PoseActionBase):
    def __init__(self, pose: Pose, rotation_representation: RotationRepresentation, gripper_command: GripperCommand):
        if (numpy.abs(pose.p)).max() > 1:  # position clipping
            pose.p = numpy.clip(pose.p, -1, 1)
        super().__init__(pose, rotation_representation, gripper_command=gripper_command)

    @override
    def to_delta_ee_pose(
        self, pinocchio_model: pinocchio.PinocchioModel, ee_link_index: int, current_pose: Pose
    ) -> RobotAction:
        return self

    @override
    def to_target_joint_position(self) -> RobotAction:
        pass


class TargetJointPositionAction(RobotAction):
    def __init__(self, target_position: numpy.ndarray, gripper_command: GripperCommand):
        self.__target_position: Tensor = torch.from_numpy(target_position).float()
        super().__init__(gripper_command=gripper_command)

    @property
    def target_position(self):
        return self.__target_position

    @override
    def to_tensor(self) -> torch.Tensor:
        return torch.hstack([self.__target_position, self._gripper_command])

    @override
    def to_delta_ee_pose(
        self, pinocchio_model: pinocchio.PinocchioModel, ee_link_index: int, current_pose: Pose
    ) -> RobotAction:
        # Just created to have a variable name, but it is irrelevant since those values will not be set as an action.
        gripper_joints_position: numpy.ndarray = numpy.array([0, 0])
        pinocchio_model.compute_forward_kinematics(numpy.hstack([self.target_position, gripper_joints_position]))
        ee_target_pose = Pose(obj=pinocchio_model.get_link_pose(ee_link_index))
        delta_ee_pose = Pose(obj=(current_pose.inv() * ee_target_pose))
        return DeltaEEPoseAction(
            pose=delta_ee_pose,
            rotation_representation=RotationRepresentation.EULER,
            gripper_command=self.gripper_command,
        )

    @override
    def to_target_joint_position(self) -> RobotAction:
        return self

    @classmethod
    def from_tensor(cls, input_tensor: torch.Tensor) -> RobotAction:
        gripper_command = GripperCommand.from_tensor(input_tensor[-1])
        return cls(target_position=input_tensor[:-1].numpy(), gripper_command=gripper_command)
