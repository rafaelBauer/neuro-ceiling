from enum import IntEnum

import torch
from overrides import override

from envs.robotactions import GripperCommand
from utils.gripperstate import GripperState
from utils.labeltoobjectpose import LabelToPoseTranslator
from utils.sceneobservation import SceneObservation
from .goal import Goal
from utils.pose import Pose


class PickPlaceObject(Goal):

    class Objective(IntEnum):
        PICK = 1
        PLACE = 2

    def __init__(self, pose: Pose, objective: Objective, label_source: torch.Tensor = torch.Tensor([])):
        """
        The constructor for PickObject class.

        Parameters:
            pose (Pose): The initial pose of the object.
        """
        self.__pose: Pose = pose
        self.__objective: PickPlaceObject.Objective = objective
        self.__label_source: torch.Tensor = label_source
        super().__init__()

    def __str__(self):
        return f"{self.__objective.name} object to/from Pose {self.__pose}"

    def __eq__(self, other: "PickPlaceObject") -> bool:
        if not isinstance(other, PickPlaceObject):
            return False
        return self.__pose.is_close(other.__pose, atol=0.001) and self.__objective == other.__objective

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
        return self.__label_source

    @classmethod
    def from_label_tensor(cls, input_tensor: torch.Tensor, current_observation: SceneObservation) -> Goal:
        # If statement only to protect against very first iteration where the current_observation is empty
        if len(current_observation.spots.values()) == 0:
            return Goal(input_tensor.size(0))
        object_poses, spots_poses = LabelToPoseTranslator.adjust_objects_and_spots_poses(current_observation)

        if current_observation.gripper_state == GripperState.OPENED or len(current_observation.objects.values()) == len(
            object_poses
        ):
            pick_place = PickPlaceObject.Objective.PICK
            pose_source = object_poses
        else:
            pick_place = PickPlaceObject.Objective.PLACE
            pose_source = spots_poses

        target_pose = LabelToPoseTranslator.get_pose_from_label(input_tensor, pose_source)

        if target_pose is not None:
            new_goal = cls(pose=target_pose, objective=pick_place, label_source=input_tensor)
        else:
            new_goal = Goal(input_tensor.size(0))

        return new_goal
