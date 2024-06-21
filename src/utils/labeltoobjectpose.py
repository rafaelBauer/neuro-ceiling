from typing import Optional

import numpy
import torch

from utils.gripperstate import GripperState
from utils.pose import Pose
from utils.sceneobservation import SceneObservation


class LabelToPoseTranslator:
    __CUBE_SIZE: float = 0.04

    @classmethod
    def get_pose_from_label(cls, current_label: torch.Tensor, pose_list: list[Pose]) -> Optional[Pose]:
        # label: [spot1/object1, spot2/object2, ..., spotN/objectN, No object/spot]
        # pose_list: [pose1, pose2, ..., poseN]
        assert (
            len(current_label) == len(pose_list) + 1
        ), "The length of the label should be equal to the length of the pose list plus one"
        target_pose: Optional[Pose] = None
        for i, pose in enumerate(pose_list):
            if current_label[i]:
                target_pose = pose
                break
        if target_pose is None:
            current_label[-1] = True
        return target_pose

    @classmethod
    def adjust_objects_and_spots_poses(cls, current_observation: SceneObservation):
        end_effector_pose = Pose(raw_euler_pose=current_observation.end_effector_pose)
        object_poses = [Pose(raw_euler_pose=raw_pose) for raw_pose in list(current_observation.objects.values())]
        spots_poses = [Pose(raw_euler_pose=raw_pose) for raw_pose in list(current_observation.spots.values())]
        cls.__adjust_pose_of_stacked_objects(object_poses, end_effector_pose, current_observation.gripper_state)
        spots_poses = cls.__adjust_available_spots_poses(spots_poses, object_poses)
        return object_poses, spots_poses

    @classmethod
    def __adjust_available_spots_poses(
        cls, spots_poses: list[Pose], object_poses: list[Pose], cube_size: float = __CUBE_SIZE
    ) -> list[Pose]:
        """
        This function computes the available places for the objects in the scene.

        It iterates over each spot in the scene and checks if there is an object on top of it.
        If there is an object on top of the spot, it replaces the spot's position with the object's position.
        This is done because the object might not be straight, and we want to place the next object on top of it.

        Args:
            spots_poses (list[Pose]): A list of poses representing the spots in the scene.
            object_poses (list[Pose]): A list of poses representing the objects in the scene.

        Returns:
            list[Pose]: A list of updated spot poses considering the positions of the objects in the scene.
        """
        __CUBE_HALF_SIZE: float = cube_size / 2

        for i, (spot_pose) in enumerate(spots_poses):
            for object_pose in object_poses:
                # Check if the X and Y positions are the same and if there is an object on top of the spot
                if spot_pose.is_same_xy_position(object_pose, atol=__CUBE_HALF_SIZE) and spot_pose.p[2] <= (
                    object_pose.p[2] + 0.01
                ):
                    # Replace the position of spot by object on top, since maybe the object is not straight
                    spots_poses[i] = Pose(p=(object_pose.p + [0, 0, cube_size]), q=object_pose.q)
        return spots_poses

    @classmethod
    def __adjust_pose_of_stacked_objects(
        cls, object_poses: list[Pose], end_effector_pose: Pose, gripper_state: GripperState, cube_size: float = 0.04
    ):
        """
        This function adjusts the poses of stacked objects in the scene.

        It first checks if the gripper is holding an object. If so, it removes this object from the list of object poses,
        since an object cannot be placed on top of itself.

        Then, it iterates over each pair of objects in the scene and checks if they have the same X and Y positions.
        If they do, it means that one object is on top of the other. In this case, it replaces the Z position of the lower
        object with the Z position of the object on top. This is done to ensure that the next object will be placed on top
        of the highest object at that X and Y position.

        Args:
            object_poses (list[Pose]): A list of poses representing the objects in the scene.
            end_effector_pose (Pose): The pose of the end effector.
            gripper_state (GripperState): The current state of the gripper.
        """
        # Remove the object that is being held
        # by the end effector, since the object cannot be placed on top of itself
        if gripper_state == GripperState.CLOSED:
            object_poses_cleaned = [
                pose for pose in object_poses if not numpy.allclose(pose.p, end_effector_pose.p, atol=0.01)
            ]
            object_poses[:] = object_poses_cleaned

        # Check if there is any object on top of another and replace the position of the lower one by the one on top
        for i in range(len(object_poses)):
            for j in range(i + 1, len(object_poses)):
                # Check if the X and Y positions are the same
                if object_poses[i].is_same_xy_position(object_poses[j], atol=cube_size):
                    if object_poses[i].p[2] < object_poses[j].p[2]:
                        # Replace the Z position of the lower object by the one on top
                        object_poses[i].p = [object_poses[i].p[0], object_poses[i].p[1], object_poses[j].p[2]]
                    else:
                        object_poses[j].p = [object_poses[j].p[0], object_poses[j].p[1], object_poses[i].p[2]]
