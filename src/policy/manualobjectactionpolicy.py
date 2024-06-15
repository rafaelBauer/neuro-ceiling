from dataclasses import dataclass, field
from typing import Optional

import numpy
import numpy as np
from overrides import override
from torch import Tensor

from envs.scene import Scene
from goal.goal import Goal
from goal.pickplaceobject import PickPlaceObject
from policy.manualpolicy import ManualPolicyConfig, ManualPolicy
from utils.gripperstate import GripperState
from utils.keyboard_observer import KeyboardObserver
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
        __scene (Scene): The scene in which the policy is applied.
        __last_goal (Goal): The last goal that was set.
    """

    def __init__(
        self, config: ManualObjectActionPolicyConfig, keyboard_observer: KeyboardObserver, scene: Scene, **kwargs
    ):
        self.__scene: Scene = scene
        self.__last_goal: Goal = Goal()
        super().__init__(config, keyboard_observer, **kwargs)
        self._keyboard_observer.subscribe_callback_to_direction(self.__key_pressed_callback)
        self.__last_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.__new_command = False
        self.__last_gripper_state: GripperState = GripperState.OPENED

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

        self.__new_command = False

        end_effector_pose = Pose(raw_euler_pose=current_observation.end_effector_pose)
        object_poses = [Pose(raw_euler_pose=raw_pose) for raw_pose in list(current_observation.objects.values())]
        self.__fix_pose_of_stacked_objects(object_poses, end_effector_pose, current_observation.gripper_state)

        if current_observation.gripper_state == GripperState.OPENED:
            pick_place = PickPlaceObject.Objective.PICK
            pose_source = object_poses
        else:
            pick_place = PickPlaceObject.Objective.PLACE
            spots_poses = [Pose(raw_euler_pose=raw_pose) for raw_pose in list(current_observation.spots.values())]
            pose_source: [Pose] = self.__compute_available_place_poses(spots_poses, object_poses)

        if self.__last_action[5] < -0.5:  # "u" key
            target_pose: Pose = pose_source[0]
        elif self.__last_action[4] < -0.5:  # "i" key
            target_pose: Pose = pose_source[1]
        elif self.__last_action[5] > 0.5:  # "o" key
            target_pose: Pose = pose_source[2]

        if target_pose is not None:
            new_goal = PickPlaceObject(pose=target_pose, objective=pick_place)
        else:
            new_goal = Goal(self.__last_goal.to_tensor().size(0))

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

    def __compute_available_place_poses(self, spots_poses: list[Pose], object_poses: list[Pose]) -> list[Pose]:
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
        for i, (spot_pose) in enumerate(spots_poses):
            for object_pose in object_poses:
                # Check if the X and Y positions are the same and if there is an object on top of the spot
                if spot_pose.is_same_xy_position(object_pose, atol=self.__CUBE_HALF_SIZE) and spot_pose.p[2] <= (
                    object_pose.p[2] + 0.01
                ):
                    # Replace the position of spot by object on top, since maybe the object is not straight
                    spots_poses[i] = Pose(p=(object_pose.p + [0, 0, self.__CUBE_SIZE]), q=object_pose.q)
        return spots_poses

    def __fix_pose_of_stacked_objects(
        self, object_poses: list[Pose], end_effector_pose: Pose, gripper_state: GripperState
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
                if object_poses[i].is_same_xy_position(object_poses[j], atol=self.__CUBE_SIZE):
                    if object_poses[i].p[2] < object_poses[j].p[2]:
                        # Replace the Z position of the lower object by the one on top
                        object_poses[i].p = [object_poses[i].p[0], object_poses[i].p[1], object_poses[j].p[2]]
                    else:
                        object_poses[j].p = [object_poses[j].p[0], object_poses[j].p[1], object_poses[i].p[2]]
