from typing import override

from envs.robotactions import GripperCommand
from .task import Task
from utils.pose import Pose


class MoveObjectToPosition(Task):
    """
    This class represents a task of moving an object from one position to another.

    Attributes:
        __object_pose (Pose): The initial pose of the object.
        __target_pose (Pose): The target pose where the object needs to be moved to.
    """

    def __init__(self, object_pose: Pose, target_pose: Pose):
        """
        The constructor for MoveObjectToPosition class.

        Parameters:
            object_pose (Pose): The initial pose of the object.
            target_pose (Pose): The target pose where the object needs to be moved.
        """
        self.__object_pose = object_pose
        self.__target_pose = target_pose
        super().__init__()

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
        pose_above_take_object = self.__object_pose.copy()
        pose_above_release_object = self.__target_pose.copy()
        # An offset of 0.1 is added to the Z coordinate to ensure that the gripper does not collide with the object
        pose_above_take_object.p = pose_above_take_object.p + [0, 0, 0.1]
        pose_above_release_object.p = pose_above_release_object.p + [0, 0, 0.1]
        return [
            (pose_above_take_object, GripperCommand.OPEN),
            (self.__object_pose, GripperCommand.OPEN),
            (self.__object_pose, GripperCommand.CLOSE),
            (pose_above_take_object, GripperCommand.CLOSE),
            (self.__target_pose, GripperCommand.CLOSE),
            (self.__target_pose, GripperCommand.OPEN),
            (pose_above_release_object, GripperCommand.OPEN),
        ]
