from typing import override

from envs.robotactions import GripperCommand
from .task import Task
from utils.pose import Pose


class MoveObjectToPosition(Task):
    def __init__(self, object_pose: Pose, target_pose: Pose):
        self.__object_pose = object_pose
        self.__target_pose = target_pose
        super().__init__()

    @override
    def get_action_sequence(self) -> list[tuple[Pose, GripperCommand]]:
        pose_above_take_object = self.__object_pose.copy()
        pose_above_release_object = self.__target_pose.copy()
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
