from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Final, override, final, Sequence

import gymnasium as gym
import numpy as np
from mani_skill.envs.sapien_env import BaseEnv

from utils.logging import log_constructor, logger
from utils.pose import Pose
from .mani_skill.neuroceilingenv import __ENV_NAME__
from .environment import BaseEnvironment, BaseEnvironmentConfig
from .robotactions import RobotAction
from .robotinfo import RobotInfo, RobotMotionInfo


@dataclass
class ManiSkillEnvironmentConfig(BaseEnvironmentConfig):
    def __init__(self):
        super().__init__("ManiSkill")
        self.headless: bool = False
        self.render_sapien: bool = True


class ManiSkillEnv(BaseEnvironment):
    __RENDER_MODE: Final[str] = "human"
    __CONTROL_MODE: Final[str] = "pd_ee_delta_pose"  # "pd_joint_pos", "pd_ee_delta_pose"

    # -------------------------------------------------------------------------- #
    # Initialization
    # -------------------------------------------------------------------------- #
    @log_constructor
    def __init__(self, config: ManiSkillEnvironmentConfig) -> None:
        super().__init__(config)
        self.__HEADLESS: bool = config.headless
        self.__render_sapien: bool = config.render_sapien
        kwargs = {
            "control_mode": self.__CONTROL_MODE,
            "render_mode": self.__RENDER_MODE,
            "reward_mode": "sparse",
        }
        self.__env: Final[BaseEnv] = gym.make(__ENV_NAME__, **kwargs)

        self.__end_effector_link_index: Final[int] = [
            link for link in self.__env.get_wrapper_attr("agent").robot.links if "hand_tcp" in link.name
        ][0].index.item()
        self.__pinocchio_model = self.__env.get_wrapper_attr("agent").robot.create_pinocchio_model()

    @override
    def start(self):
        logger.info("Starting ManiSkill environment {}", __ENV_NAME__)
        self.reset()
        logger.info("ManiSkill environment {} successfully started", __ENV_NAME__)

    # -------------------------------------------------------------------------- #
    # Observation
    # -------------------------------------------------------------------------- #

    # -------------------------------------------------------------------------- #
    # Visualization
    # -------------------------------------------------------------------------- #
    # def render(self) -> None:
    #     if not self.__HEADLESS:
    #         if self.__render_sapien:
    #             self.__env.render_human()
    #         else:
    #             obs = self.__env.render_cameras()
    #             # self.cam_win_title
    #             cv2.imshow("test", obs)
    #             cv2.waitKey(1)

    # -------------------------------------------------------------------------- #
    # Reset
    # -------------------------------------------------------------------------- #
    @override
    def reset(self):
        super().reset()
        self.__env.reset()

    @override
    def reset_joint_pose(self) -> None:
        pass

    @override
    def close(self):
        pass

    # -------------------------------------------------------------------------- #
    # Step
    # -------------------------------------------------------------------------- #
    @override
    def _step(
        self,
        action: RobotAction,
        postprocess: bool = True,
        delay_gripper: bool = True,
        scale_action: bool = True,
    ) -> tuple[dict, float, bool, dict]:
        """
        Perform a single step in the environment.

        Args:
            action (np.ndarray[Literal[7]]): The action to take in the environment. (x,y,z,rx,ry,rz,gripper)
            postprocess (bool, optional): Whether to postprocess the action. Defaults to True.
            delay_gripper (bool, optional): Whether to delay the gripper. Defaults to True.
            scale_action (bool, optional): Whether to scale the action. Defaults to True.

        Returns:
            tuple[dict, float, bool, dict]: A tuple containing the next observation, reward, done flag, and additional info.
        """

        if self.__CONTROL_MODE == "pd_joint_pos":
            robot_raw_action = action.to_target_joint_position().get_raw_action()
        elif self.__CONTROL_MODE == "pd_ee_delta_pose":
            motion_info = self.get_robot_motion_info()
            robot_raw_action = action.to_delta_ee_pose(
                self.__pinocchio_model, self.__end_effector_link_index, motion_info.current_ee_pose
            ).get_raw_action()
        else:
            raise NotImplementedError(f"Action for control mode {self.__CONTROL_MODE} not implemented")

        next_obs, reward, done, _, info = self.__env.step(robot_raw_action)

        obs = next_obs

        self.__env.render()

        return obs, reward, done, info

    # -------------------------------------------------------------------------- #
    # Info
    # -------------------------------------------------------------------------- #
    @override
    def get_robot_info(self) -> final(RobotInfo):
        LINK_NAMES: Final[Sequence[str]] = [
            link.get_name() for link in self.__env.get_wrapper_attr("agent").robot.get_links()
        ]
        JOINT_NAMES: Final[Sequence[str]] = [
            joint.get_name() for joint in self.__env.get_wrapper_attr("agent").robot.get_active_joints()
        ]
        URDF_PATH: Final[Path] = Path(self.__env.get_wrapper_attr("agent").urdf_path)
        SRDF_PATH: Final[Path] = URDF_PATH.with_suffix(".srdf")
        END_EFFECTOR_LINK_NAME: Final[str] = LINK_NAMES[self.__end_effector_link_index]

        return RobotInfo(
            urdf_path=URDF_PATH,
            srdf_path=SRDF_PATH,
            links=LINK_NAMES,
            joints=JOINT_NAMES,
            end_effector_link=END_EFFECTOR_LINK_NAME,
        )

    @override
    def get_robot_motion_info(self) -> final(RobotMotionInfo):
        # Take index 0 because it is of shape (1,9), and we want a dim0  array.
        end_effector_pose = (
            self.__env.get_wrapper_attr("agent").robot.get_links()[self.__end_effector_link_index].pose.sp
        )  # W.R.T what??
        robot_pose = self.__env.get_wrapper_attr("agent").robot.pose.sp
        end_effector_pose_wrt_base = robot_pose.inv() * end_effector_pose
        return RobotMotionInfo(
            current_qpos=self.__env.get_wrapper_attr("agent").robot.get_qpos().cpu()[0],
            current_ee_pose=Pose(obj=end_effector_pose_wrt_base),
        )
