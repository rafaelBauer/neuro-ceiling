from overrides import override
from tensordict import tensorclass

from envs.robotactions import GripperCommand
from .goal import Goal
from utils.pose import Pose


class MoveObjectToPosition(Goal):
    """
    This class represents a goal of moving an object from one position to another.

    Attributes:
        __object_pose (Pose): The initial pose of the object.
        __target_pose (Pose): The target pose where the object needs to be moved to.
    """
    __object_pose: Pose
    __target_pose: Pose

    def __init__(self, object_pose: Pose, target_pose: Pose):
        """
        The constructor for MoveObjectToPosition class.

        Parameters:
            object_pose (Pose): The initial pose of the object.
            target_pose (Pose): The target pose where the object needs to be moved.
        """
        self.__object_pose: Pose = object_pose
        self.__target_pose: Pose = target_pose
        super().__init__()

    def __str__(self):
        return f"MoveObjectToPosition from Pose {self.__object_pose} to {self.__target_pose}"

    def __eq__(self, other: 'MoveObjectToPosition') -> bool:
        if not isinstance(other, MoveObjectToPosition):
            return False
        return self.__object_pose == other.__object_pose and self.__target_pose == other.__target_pose

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
