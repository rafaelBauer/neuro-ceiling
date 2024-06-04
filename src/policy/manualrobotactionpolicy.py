from dataclasses import dataclass, field
from typing import override

import numpy
from torch import Tensor

from envs.robotactions import GripperCommand, DeltaEEPoseAction
from policy.manualpolicy import ManualPolicyConfig, ManualPolicy
from utils.keyboard_observer import KeyboardObserver
from utils.pose import Pose, RotationRepresentation


@dataclass(kw_only=True)
class ManualRobotActionPolicyConfig(ManualPolicyConfig):
    """
    Configuration class for ManualRobotActionPolicy. Inherits from ManualPolicy.
    """

    _POLICY_TYPE: str = field(init=False, default="ManualRobotActionPolicy")


class ManualRobotActionPolicy(ManualPolicy):
    def __init__(self, config: ManualRobotActionPolicyConfig, keyboard_observer: KeyboardObserver, **kwargs):
        super().__init__(config, keyboard_observer, **kwargs)

    @override
    def specific_forward(self, action: numpy.array) -> Tensor:
        if action[-1] < 0:
            gripper_command = GripperCommand.CLOSE
        else:
            gripper_command = GripperCommand.OPEN
        delta_pose = Pose(p=action[:3], euler=action[3:])
        return DeltaEEPoseAction(
            delta_pose, rotation_representation=RotationRepresentation.EULER, gripper_command=gripper_command
        )
