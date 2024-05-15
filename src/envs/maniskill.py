from dataclasses import dataclass
from typing import Literal, Optional, Final, override

import gymnasium as gym
import numpy as np
from mani_skill.envs.sapien_env import BaseEnv

from .mani_skill.neuroceilingenv import __ENV_NAME__
from .environment import BaseEnvironment, BaseEnvironmentConfig


@dataclass
class ManiSkillEnvironmentConfig(BaseEnvironmentConfig):
    def __init__(self):
        super().__init__("ManiSkill")
        self.headless: bool = False
        self.render_sapien: bool = True


class ManiSkillEnv(BaseEnvironment):
    __RENDER_MODE: Final[str] = "human"
    __CONTROL_MODE: Final[str] = "pd_ee_delta_pose"

    # -------------------------------------------------------------------------- #
    # Initialization
    # -------------------------------------------------------------------------- #
    def __init__(self, config: ManiSkillEnvironmentConfig) -> None:
        super().__init__(config)
        self.__HEADLESS: bool = config.headless
        self.__render_sapien: bool = config.render_sapien
        self.__env: Optional[BaseEnv] = None

    @override
    def start(self):
        kwargs = {
            "control_mode": self.__CONTROL_MODE,
            "render_mode": self.__RENDER_MODE,
            "reward_mode": "sparse",
        }
        self.__env = gym.make(__ENV_NAME__, **kwargs)
        self.reset()

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
        action: np.ndarray[Literal[7]],
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

        next_obs, reward, done, _, info = self.__env.step(action)

        obs = next_obs

        self.__env.render()

        return obs, reward, done, info
