from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from omegaconf import MISSING

from utils.geometry_np import axis_angle_to_quaternion, quaternion_to_axis_angle
from utils.observation import SceneObservation


# from utils.geometry_np import (
#     axis_angle_to_quaternion,
#     euler_to_quaternion,
#     normalize_quaternion,
#     quat_real_first_to_real_last,
#     quaternion_to_axis_angle,
# )
# from utils.observation import SceneObservation


def squash(array, order=20):
    # map to [-1, 1], but more linear than tanh
    return np.sign(array) * np.power(
        np.tanh(np.power(np.power(array, 2), order / 2)), 1 / order
    )


class GripperPlot:
    def __init__(self, headless):
        self.headless = headless

        if headless:
            return

        self.displayed_gripper = 0.9

        self.fig = plt.figure()

        ax = self.fig.add_subplot(111)
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)

        horizontal_patch = plt.Rectangle((-1, 0), 2, 0.6)
        self.left_patch = plt.Rectangle((-0.9, -1), 0.4, 1, color="black")
        self.right_patch = plt.Rectangle((0.5, -1), 0.4, 1, color="black")

        ax.add_patch(horizontal_patch)
        ax.add_patch(self.left_patch)
        ax.add_patch(self.right_patch)

        self.fig.canvas.draw()

        plt.show(block=False)

        plt.pause(0.1)

        for _ in range(2):
            self.set_data(0)
            plt.pause(0.1)
            self.set_data(1)
            plt.pause(0.1)

    def set_data(self, new_state: float) -> None:
        """
        Set the gripper plot to the given gripper state.

        Parameters
        ----------
        new_state : float
            The new gripper state, either 0.9 (open) or -0.9 (closed).
        """
        if self.headless or self.displayed_gripper == new_state:
            return

        if new_state == 0.9:
            self.displayed_gripper = 0.9
            self.left_patch.set_xy((-0.9, -1))
            self.right_patch.set_xy((0.5, -1))

        elif new_state == -0.9:
            self.displayed_gripper = -0.9
            self.left_patch.set_xy((-0.4, -1))
            self.right_patch.set_xy((0, -1))

        self.fig.canvas.draw()

        plt.pause(0.01)

        return

    def reset(self) -> None:
        self.set_data(1)


class BaseEnvironmentConfig:
    def __init__(self, ENV_TYPE : str):
        self.__ENV_TYPE: str = ENV_TYPE

    @property
    def env_type(self) -> str:
        return self.__ENV_TYPE


class BaseEnvironment(ABC):
    def __init__(self, config: BaseEnvironmentConfig) -> None:
        self.CONFIG: BaseEnvironmentConfig = config

        self.do_postprocess_actions = False     # config.postprocess_actions
        self.do_scale_action = False            # config.scale_action
        self.do_delay_gripper = False           # config.delay_gripper

        image_size = (256, 256)                 # config.image_size

        self.image_height, self.image_width = image_size

        self.gripper_plot = GripperPlot(False) # (not config.gripper_plot)
        self.gripper_open = 0.9
        self.gripper_deque = None

    def reset(self, **kwargs) -> None:
        """
        Reset the environment to a new episode. In the BaseEnvironment, this
        only resets the gripper plot.
        :param kwargs: Not used by base
        :return:
        """

        if self.gripper_plot:
            self.gripper_plot.reset()

        self.gripper_open = 0.9
        # self.gripper_deque = deque([0.9] * self.queue_length, maxlen=self.queue_length)

    def step(self, action: np.ndarray) -> tuple[SceneObservation, float, bool, dict]:
        """
        Postprocess the action and execute it in the environment.
        Simple wrapper around _step, that provides the kwargs for
        postprocessing from self.config.

        Parameters
        ----------
        action : np.ndarray
            The raw action predicted by a policy.

        Returns
        -------
        tuple[SceneObservation, float, bool, dict]
            The observation, reward, done flag and info dict.
        """

        return self._step(
            action,
            postprocess=self.do_postprocess_actions,
            delay_gripper=self.do_delay_gripper,
            scale_action=self.do_scale_action,
        )

    @abstractmethod
    def _step(
            self,
            action: np.ndarray,
            postprocess: bool = True,
            delay_gripper: bool = True,
            scale_action: bool = True,
    ) -> tuple[SceneObservation, float, bool, dict]:
        """
        Postprocess the action and execute it in the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def start(self) -> None:
        """
        Starts the environment. In the real word would open the connections and do everything needed to allow the
        interactions. In the simulation, it creates the environment.

        :return:
        """

    @abstractmethod
    def close(self) -> None:
        """
        Gracefully close the environment.
        """
        raise NotImplementedError

    def postprocess_action(
            self,
            action: np.ndarray,
            prediction_is_quat: bool = True,
            scale_action: bool = False,
            delay_gripper: bool = False,
            trans_scale: float | None = None,
            rot_scale: float | None = None,
    ) -> np.ndarray:
        """
        Postprocess the action predicted by the policy for the action space of
        the environment.

        Parameters
        ----------
        action : np.ndarray[(7,), np.float32]
            Original action predicted by the policy
            Concatenation of delta_position, delta rotation, gripper action.
            Delta rotation can be axis angle (NN) or Quaternion (GMM).
        scale_action : bool, optional
            Whether to scale the position and rotation action, by default False
        delay_gripper : bool, optional
            Whether to delay the gripper, by default False
        trans_scale : float | None, optional
            The scaling for the translation action,
            by default self._delta_pos_scale
        rot_scale : float | None, optional
            The scaling for the rotation (applied to the Euler angles),
            by default self._delta_angle_scale

        Returns
        -------
        np.ndarray
            _description_
        """
        if trans_scale is None:
            trans_scale = self._delta_pos_scale
        if rot_scale is None:
            rot_scale = self._delta_angle_scale

        rot_dim = 4 if prediction_is_quat else 3

        delta_position, delta_rot, gripper = np.split(action, [3, 3 + rot_dim])

        if prediction_is_quat:
            delta_rot_axis_angle = quaternion_to_axis_angle(delta_rot)
        else:
            delta_rot_axis_angle = delta_rot

        if scale_action:
            delta_position = delta_position * trans_scale
            delta_rot_axis_angle = delta_rot_axis_angle * rot_scale

        delta_rot_quat = axis_angle_to_quaternion(delta_rot_axis_angle)

        delta_rot_env = self.postprocess_quat_action(delta_rot_quat)

        if delay_gripper:
            gripper = [self.delay_gripper(gripper)]

        return np.concatenate((delta_position, delta_rot_env, gripper))


    def postprocess_quat_action(self, quaternion: np.ndarray) -> np.ndarray:
        """
        Postprocess the rotation action predicted by the policy for the action
        space of the environment.

        Parameters
        ----------
        quaternion : np.ndarray
            Quaternion action predicted by the policy (real first).

        Returns
        -------
        np.ndarray
            The rotation action in the action space of the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_joint_pose(self) -> None:
        raise NotImplementedError("Need to implement in child class.")
