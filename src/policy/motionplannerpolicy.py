from dataclasses import dataclass, field
from enum import IntEnum
from typing import override, Final

import numpy as np
import torch
from torch import Tensor
from mplib import Planner, Pose

from envs import BaseEnvironment
from envs.robotinfo import RobotInfo
from policy.policy import PolicyBaseConfig, PolicyBase
from utils.logging import logger, log_constructor


class GripperCommand(IntEnum):
    """
    An enumeration representing the commands that can be sent to the gripper.

    Attributes
    ----------
    OPEN : int
       A command to open the gripper. The value of this command is 1.
    CLOSE : int
       A command to close the gripper. The value of this command is -1.
    """

    OPEN = 1
    CLOSE = -1


@dataclass(kw_only=True)
class MotionPlannerPolicyConfig(PolicyBaseConfig):
    """
    A dataclass representing the configuration for the MotionPlannerPolicy.

    This class inherits from PolicyBaseConfig and adds a specific policy type for the MotionPlannerPolicy.

    Attributes
    ----------
    _POLICY_TYPE : str
        A string representing the type of policy. This is set to "MotionPlannerPolicy" and
        cannot be changed during initialization.
    """

    _POLICY_TYPE: str = field(init=False, default="MotionPlannerPolicy")


class MotionPlannerPolicy(PolicyBase):
    """
    A class representing the Motion Planner Policy.

    This is a deterministic policy used to abstract the low level control of the robot.

    Attributes
    ----------
    __config : MotionPlannerPolicyConfig
        The configuration for the MotionPlannerPolicy.
    __path_planner : Planner
        The path planner used to plan the robot's movements.
    __gripper_command : GripperCommand
        The current command for the gripper.
    __last_action : Tensor
        The last action taken by the robot.
    __current_path : list[np.ndarray]
        The current path that the robot is following.
    """

    @log_constructor
    def __init__(self, config: MotionPlannerPolicyConfig, environment: BaseEnvironment, **kwargs):
        """
        Initializes the MotionPlannerPolicy with the given configuration and environment.

        Parameters
        ----------
        config : MotionPlannerPolicyConfig
            The configuration for the MotionPlannerPolicy.
        environment : BaseEnvironment
            The environment in which the robot is operating.
        """
        self.__config = config

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

        super().__init__(config, **kwargs)

        self.__gripper_command: GripperCommand = GripperCommand.OPEN

        current_motion_info = environment.get_robot_motion_info()

        # At the current pose and with the gripper opened
        self.__last_action: Tensor = torch.from_numpy(
            np.hstack([current_motion_info.current_ee_pose, self.__gripper_command])
        )

        initial_pose: Pose = Pose(p=np.array([0.615, 0.0, 0.02]), q=np.array([0, 1, 0, 0]))
        current_qpos = current_motion_info.current_qpos.numpy()

        self.__gripper_command: GripperCommand = GripperCommand.OPEN
        self.__current_path: list[np.ndarray] = self.__plan_to_pose(current_qpos, initial_pose)
        logger.info("Initialized planner to initial pose {} with gripper {}", initial_pose, self.__gripper_command.name)

    @override
    def update(self):
        """
        Updates the policy. This method is currently not implemented.
        """
        pass

    @override
    def forward(self, states: Tensor) -> Tensor:
        """
        Samples actions to be taken by the robot based on the current path.

        Parameters
        ----------
        states : Tensor
            The current state of the robot.

        Returns
        -------
        Tensor
            The action to be taken by the robot.
        """
        if self.__current_path:
            action: np.ndarray = np.hstack([self.__current_path.pop(0), self.__gripper_command])
            self.__last_action = torch.from_numpy(action)
        return self.__last_action

    def __plan_to_pose(self, current_qpos: np.ndarray, target_pose: Pose, time_step=0.05) -> list[np.ndarray]:
        """
        Plans a path from the current pose to the target end-effector pose.

        Parameters
        ----------
        current_qpos : np.ndarray
            The current qpos (controllable joint positions) of the robot.
        target_pose : Pose
            The target pose for the end-effector robot.
        time_step : float, optional
            The time step for the path planning, by default 0.05.

        Returns
        -------
        list[np.ndarray]
            The positions of the planned path.
        """
        plan = self.__path_planner.plan_screw(target_pose, current_qpos, time_step=time_step)

        if not plan["status"] == "Success":
            logger.error("Could not plan path. Current pose {} -> Target pose {}", current_qpos, target_pose)
            return []
        return plan.pop("position").tolist()
