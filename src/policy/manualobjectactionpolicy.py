from dataclasses import dataclass, field
from typing import override, Optional

import numpy
from torch import Tensor

from envs.scene import Scene
from goal.goal import Goal
from goal.movetoposition import MoveObjectToPosition
from policy.manualpolicy import ManualPolicyConfig, ManualPolicy
from utils.keyboard_observer import KeyboardObserver
from utils.pose import Pose


@dataclass(kw_only=True)
class ManualObjectActionPolicyConfig(ManualPolicyConfig):
    """
    Configuration class for ManualObjectActionPolicyConfig. Inherits from ManualPolicy.
    """

    _POLICY_TYPE: str = field(init=False, default="ManualObjectActionPolicy")


class ManualObjectActionPolicy(ManualPolicy):
    def __init__(
        self, config: ManualObjectActionPolicyConfig, keyboard_observer: KeyboardObserver, scene: Scene, **kwargs
    ):
        self.__scene: Scene = scene
        self.__last_goal: Goal = Goal()
        super().__init__(config, keyboard_observer, **kwargs)

    @override
    def specific_forward(self, action: numpy.array) -> Tensor:
        object_pose: Optional[Pose] = None
        target_pose: Optional[Pose] = None

        if action[3] > 0.5:  # "l" key
            target_pose = self.__scene.spots[2].pose.copy()
        elif action[3] < -0.5:  # "j" key
            target_pose = self.__scene.spots[0].pose.copy()

        if action[4] > 0.5:  # "k" key
            target_pose = self.__scene.spots[1].pose.copy()
        elif action[4] < -0.5:  # "i" key
            object_pose: Pose = self.__scene.objects[1].pose

        if action[5] > 0.5:  # "o" key
            object_pose: Pose = self.__scene.objects[2].pose
        elif action[5] < -0.5:  # "u" key
            object_pose: Pose = self.__scene.objects[0].pose

        if object_pose is not None and target_pose is not None:
            target_pose.p = target_pose.p + [0, 0, 0.04]
            self.__last_goal = MoveObjectToPosition(object_pose, target_pose)
        return self.__last_goal
