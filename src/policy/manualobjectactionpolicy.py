from dataclasses import dataclass, field
from typing import Optional

import numpy
from overrides import override
from tensordict import TensorDict
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

    @override
    def specific_forward(self, action: numpy.array, current_observation: SceneObservation) -> Tensor:
        target_pose: Optional[Pose] = None

        if len(current_observation.spots.values()) == 0:
            return self.__last_goal

        # TODO: This should be checked only if there is a new action to be taken

        end_effector_pose = Pose(raw_euler_pose=current_observation.end_effector_pose)
        object_poses = [Pose(raw_euler_pose=raw_pose) for raw_pose in list(current_observation.objects.values())]
        # Check if there is any object on top of another and replace the position of the lower one by the one on top
        for i in range(len(object_poses)):
            for j in range(i + 1, len(object_poses)):
                # Check if the X and Y positions are the same and only if the
                if numpy.isclose(object_poses[i].p[0], object_poses[j].p[0], atol=0.04) and numpy.isclose(
                    object_poses[i].p[1], object_poses[j].p[1], atol=0.04
                ):
                    if object_poses[i].p[2] < object_poses[j].p[2]:
                        # Replace the Z position of the lower object by the one on top
                        object_poses[i].p = [object_poses[i].p[0], object_poses[i].p[1], object_poses[j].p[2]]
                    else:
                        object_poses[j].p = [object_poses[j].p[0], object_poses[j].p[1], object_poses[i].p[2]]

        logger.debug("Current gripper state is {}", current_observation.gripper_state.name)

        if current_observation.gripper_state == GripperState.OPENED:
            pick_place = PickPlaceObject.Objective.PICK
            pose_source = object_poses
        else:
            pick_place = PickPlaceObject.Objective.PLACE
            spots_poses = [Pose(raw_euler_pose=raw_pose) for raw_pose in list(current_observation.spots.values())]

            # Remove the object that is being held by the end effector, since the object should not affect the spot
            # poses
            for i, (object_pose) in enumerate(object_poses):
                if numpy.allclose(object_pose.p, end_effector_pose.p, atol=0.01):
                    object_poses.pop(i)

            for i, (spot_pose) in enumerate(spots_poses):
                for object_pose in object_poses:
                    # Check if the X and Y positions are the same and only if the
                    if (
                        numpy.isclose(spot_pose.p[0], object_pose.p[0], atol=0.02)
                        and numpy.isclose(spot_pose.p[1], object_pose.p[1], atol=0.02)
                        and spot_pose.p[2] <= (object_pose.p[2] + 0.01)
                    ):
                        # Replace the position of spot by object on top, since maybe the object is not straight
                        spots_poses[i] = Pose(p=(object_pose.p + [0, 0, 0.04]), q=object_pose.q)

            pose_source: [Pose] = spots_poses

        if action[5] < -0.5:  # "u" key
            target_pose: Pose = pose_source[0]
        elif action[4] < -0.5:  # "i" key
            target_pose: Pose = pose_source[1]
        elif action[5] > 0.5:  # "o" key
            target_pose: Pose = pose_source[2]

        if target_pose is not None:
            self.__last_goal = PickPlaceObject(pose=target_pose, objective=pick_place)
        return self.__last_goal

    @override
    def episode_finished(self):
        self.__last_goal = Goal()
