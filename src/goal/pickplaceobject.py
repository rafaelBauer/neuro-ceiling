from enum import IntEnum

import torch
from overrides import override

from envs.robotactions import GripperCommand
from .goal import Goal
from utils.pose import Pose, RotationRepresentation


class PickPlaceObject(Goal):

    class Objective(IntEnum):
        PICK = 1
        PLACE = 2

    def __init__(self, pose: Pose, objective: Objective):
        """
        The constructor for PickObject class.

        Parameters:
            pose (Pose): The initial pose of the object.
        """
        self.__pose: Pose = pose
        self.__objective: PickPlaceObject.Objective = objective
        super().__init__()

    def __str__(self):
        return f"{self.__objective.name} object to/from Pose {self.__pose}"

    def __eq__(self, other: "PickPlaceObject") -> bool:
        if not isinstance(other, PickPlaceObject):
            return False
        return self.__pose == other.__pose and self.__objective == other.__objective

    @override
    def get_action_sequence(self) -> list[tuple[Pose, GripperCommand]]:
        """
        This method generates a sequence of poses as well as gripper commands to move the object from its initial
        pose to the target pose.

        Returns:
            list[tuple[Pose, GripperCommand]]: A list of tuples where each tuple contains a Pose and a GripperCommand.
            The Pose represents the pose of the gripper and the GripperCommand represents the action of the gripper,
            either open or close.
        """
        pose_above = self.__pose.copy()
        # An offset of 0.1 is added to the Z coordinate to ensure that the gripper does not collide with the object
        pose_above.p = pose_above.p + [0, 0, 0.1]
        if self.__objective == PickPlaceObject.Objective.PICK:
            first_gripper_command: GripperCommand = GripperCommand.OPEN
            second_gripper_command: GripperCommand = GripperCommand.CLOSE
        else:  # self.__objective == PickPlaceObject.Objective.PLACE:
            first_gripper_command: GripperCommand = GripperCommand.CLOSE
            second_gripper_command: GripperCommand = GripperCommand.OPEN

        return [
            (pose_above, first_gripper_command),
            (self.__pose, first_gripper_command),
            (self.__pose, second_gripper_command),
            (pose_above, second_gripper_command),
        ]

    @override
    def to_tensor(self):
        return torch.hstack(
            [
                torch.tensor(self.__objective),
                self.__pose.to_tensor(RotationRepresentation.EULER),
            ]
        )

    @classmethod
    def from_tensor(cls, input_tensor: torch.Tensor) -> Goal:
        if 0.5 < input_tensor[0] <= 1.5:
            objective = PickPlaceObject.Objective.PICK
        elif 1.5 < input_tensor[0] <= 2.5:
            objective = PickPlaceObject.Objective.PLACE
        else:
            return Goal(7)
        input_tensor[4:] = torch.Tensor([3.1415927, 0, 0])
        return cls(pose=Pose(raw_euler_pose=input_tensor[1:].detach().numpy()), objective=objective)
