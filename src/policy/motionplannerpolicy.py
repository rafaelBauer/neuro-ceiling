from dataclasses import dataclass, field
from typing import override, Final

import numpy as np
from torch import Tensor
from mplib import Planner

from envs import BaseEnvironment
from envs.robotinfo import RobotInfo
from policy.policy import PolicyBaseConfig, PolicyBase
from utils.logging import logger, log_constructor


@dataclass(kw_only=True)
class MotionPlannerPolicyConfig(PolicyBaseConfig):
    _POLICY_TYPE: str = field(init=False, default="MotionPlannerPolicy")


class MotionPlannerPolicy(PolicyBase):
    @log_constructor
    def __init__(self, config: MotionPlannerPolicyConfig, environment: BaseEnvironment, **kwargs):
        self.__config = config
        # self.__environment = environment

        ROBOT_INFO: Final[RobotInfo] = environment.get_robot_info()

        self.__path_planner = Planner(
            urdf=str(ROBOT_INFO.urdf_path),
            srdf=str(ROBOT_INFO.srdf_path),
            user_link_names=ROBOT_INFO.links,
            user_joint_names=ROBOT_INFO.joints,
            move_group=ROBOT_INFO.end_effector_link,
            joint_vel_limits=np.ones(7),
            joint_acc_limits=np.ones(7),
        )

        current_motion_info = environment.get_robot_motion_info()
        target_pose: Tensor = current_motion_info.current_end_effector_pose.raw_pose[0].clone()
        target_pose[0] = 0.2
        self.__current_path = self.__plan_to_pose(current_motion_info.current_end_effector_pose.raw_pose, target_pose)
        super().__init__(config, **kwargs)

    @override
    def update(self):
        pass

    @override
    def forward(self, states: Tensor) -> Tensor:
        return self.__current_path.pop()

    def __plan_to_pose(self, current_pose: Tensor, target_pose: Tensor, time_step=(1 / 250)):
        plan = self.__path_planner.plan_qpos_to_pose(target_pose, current_pose, time_step=time_step)

        if not plan["status"] == "Success":
            logger.error("Could not plan path. Current pose {} -> Target pose {}", current_pose, target_pose)
        return plan
