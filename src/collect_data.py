import argparse
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from loguru import logger
from omegaconf import MISSING, DictConfig, OmegaConf, SCMode
from tqdm.auto import tqdm

import utils.logging  # noqa

from envs import create_environment  # , Environment
from envs.environment import BaseEnvironmentConfig

# from policy import PolicyEnum, get_policy_class
from utils.argparse import get_config_from_args
from utils.config import ConfigBase
from utils.keyboard_observer import KeyboardObserver


# from utils.misc import (
#     DataNamingConfig,
#     get_dataset_name,
#     get_full_task_name,
#     loop_sleep,
# )


# from utils.random import configure_seeds


@dataclass
class Config(ConfigBase):
    # n_episodes: int
    # sequence_len: int | None
    #
    # data_naming: DataNamingConfig
    # dataset_config: SceneDatasetConfig

    # env: Environment
    env_config: BaseEnvironmentConfig

    # policy: PolicyEnum
    # policy_config: Any

    # pretraining_data: bool = MISSING

    # horizon: int | None = 300  # None


def parse_args() -> DictConfig:
    # parser = argparse.ArgumentParser(description='Program meant to be used to XXXXX')
    # parser.add_argument("-e", "--env-id", help='The name of the environment to execute', type=str, required=True)
    # parser.add_argument("-o", "--obs-mode", type=str)
    # parser.add_argument("--reward-mode", type=str)
    # parser.add_argument("-c", "--control-mode", type=str, default="pd_ee_delta_pose")
    # parser.add_argument("--render-mode", type=str, default="cameras")
    # parser.add_argument("--enable-sapien-viewer", action="store_true")
    # parser.add_argument("--record-dir", type=str)
    # args, opts = parser.parse_known_args()
    #
    # # Parse env kwargs
    # print("opts:", opts)
    # eval_str = lambda x: eval(x[1:]) if x.startswith("@") else x
    # env_kwargs = dict((x, eval_str(y)) for x, y in zip(opts[0::2], opts[1::2]))
    # print("env_kwargs:", env_kwargs)
    # args.env_kwargs = env_kwargs

    extra_args = (
        {
            "name": "--pretraining",
            "action": "store_true",
            "help": "Whether the data is for pretraining. Used to name the dataset.",
        },
    )
    args, dict_config = get_config_from_args(
        "Program meant to be used to XXXXX", data_load=False, extra_args=extra_args
    )
    dict_config = complete_config(args, dict_config)

    config = OmegaConf.to_container(dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE)
    return config


def complete_config(args: argparse.Namespace, config: DictConfig) -> DictConfig:
    # config.pretraining_data = args.pretraining
    #
    # config.env_config.task = config.data_naming.task
    # config.dataset_config.data_root = config.data_naming.data_root
    #
    # config.data_naming.feedback_type = get_dataset_name(config)

    return config


def main() -> None:
    np.set_printoptions(suppress=True, precision=3)
    config = parse_args()

    # Env = import_env(config.env_config)

    # Policy = get_policy_class(config.policy.value)

    # task_name = get_full_task_name(config)  # type: ignore

    # assert config.data_naming.data_root is not None

    # save_path = pathlib.Path(config.data_naming.data_root) / task_name

    # if not save_path.is_dir():
    #     logger.warning(
    #         "Creating save path. This should only be needed for " "new tasks."
    #     )
    #     save_path.mkdir(parents=True)

    env = create_environment(config.env_config)

    keyboard_obs = KeyboardObserver()

    # replay_memory = SceneDataset(
    #     allow_creation=True,
    #     config=config.dataset_config,
    #     data_root=save_path / config.data_naming.feedback_type,
    # )

    env.start()

    # policy = Policy(config.policy_config, env=env, keyboard_obs=keyboard_obs)

    #    obs = env.reset()

    time.sleep(5)

    logger.info("Go!")

    episodes_count = 0
    timesteps = 0
    lstm_state = None

    try:
        while True:
            action: np.array = np.zeros(7)
            env.step(action)
            #
            # # -------------------------------------------------------------------------- #
            # # Interaction
            # # -------------------------------------------------------------------------- #
            # # Input
            # key = opencv_viewer.imshow(render_frame)
            #
            # if has_base:
            #     assert args.control_mode in ["base_pd_joint_vel_arm_pd_ee_delta_pose"]
            #     base_action = np.zeros([4])  # hardcoded
            # else:
            #     base_action = np.zeros([0])
            #
            # # Parse end-effector action
            # if ("pd_ee_delta_pose" in args.control_mode
            #     or "pd_ee_target_delta_pose" in args.control_mode):
            #     ee_action = np.zeros([6])
            # elif (
            #         "pd_ee_delta_pos" in args.control_mode
            #         or "pd_ee_target_delta_pos" in args.control_mode
            # ):
            #     ee_action = np.zeros([3])
            # else:
            #     raise NotImplementedError(args.control_mode)
            #
            # # Base
            # if has_base:
            #     if key == "w":  # forward
            #         base_action[0] = 1
            #     elif key == "s":  # backward
            #         base_action[0] = -1
            #     elif key == "a":  # left
            #         base_action[1] = 1
            #     elif key == "d":  # right
            #         base_action[1] = -1
            #     elif key == "q":  # rotate counter
            #         base_action[2] = 1
            #     elif key == "e":  # rotate clockwise
            #         base_action[2] = -1
            #     elif key == "z":  # lift
            #         base_action[3] = 1
            #     elif key == "x":  # lower
            #         base_action[3] = -1
            #
            # # End-effector
            # if num_arms > 0:
            #     # Position
            #     if key == "i":  # +x
            #         ee_action[0] = EE_ACTION
            #     elif key == "k":  # -x
            #         ee_action[0] = -EE_ACTION
            #     elif key == "j":  # +y
            #         ee_action[1] = EE_ACTION
            #     elif key == "l":  # -y
            #         ee_action[1] = -EE_ACTION
            #     elif key == "u":  # +z
            #         ee_action[2] = EE_ACTION
            #     elif key == "o":  # -z
            #         ee_action[2] = -EE_ACTION
            #
            #     # Rotation (axis-angle)
            #     if key == "1":
            #         ee_action[3:6] = (1, 0, 0)
            #     elif key == "2":
            #         ee_action[3:6] = (-1, 0, 0)
            #     elif key == "3":
            #         ee_action[3:6] = (0, 1, 0)
            #     elif key == "4":
            #         ee_action[3:6] = (0, -1, 0)
            #     elif key == "5":
            #         ee_action[3:6] = (0, 0, 1)
            #     elif key == "6":
            #         ee_action[3:6] = (0, 0, -1)
            #
            # # Gripper
            # if has_gripper:
            #     if key == "f":  # open gripper
            #         gripper_action = 1
            #     elif key == "g":  # close gripper
            #         gripper_action = -1
            #
            # # # Other functions
            # # if key == "0":  # switch to SAPIEN viewer
            # #     render_wait()
            # # elif key == "r":  # reset env
            # #     obs, _ = env.reset()
            # #     gripper_action = 1
            # #     after_reset = True
            # #     continue
            # # elif key == None:  # exit
            # #     break
            #
            # # Visualize observation
            # # if key == "v":
            # #     if "rgbd" in env.obs_mode:
            # #         from itertools import chain
            # #
            # #         from mani_skill2.utils.visualization.misc import (
            # #             observations_to_images,
            # #             tile_images,
            # #         )
            # #
            # #         images = list(
            # #             chain(*[observations_to_images(x) for x in obs["image"].values()])
            # #         )
            # #         render_frame = tile_images(images)
            # #         opencv_viewer.imshow(render_frame)
            # #     elif "pointcloud" in env.obs_mode:
            # #         import trimesh
            # #
            # #         xyzw = obs["pointcloud"]["xyzw"]
            # #         mask = xyzw[..., 3] > 0
            # #         rgb = obs["pointcloud"]["rgb"]
            # #         if "robot_seg" in obs["pointcloud"]:
            # #             robot_seg = obs["pointcloud"]["robot_seg"]
            # #             rgb = np.uint8(robot_seg * [11, 61, 127])
            # #         trimesh.PointCloud(xyzw[mask, :3], rgb[mask]).show()
            #
            # # -------------------------------------------------------------------------- #
            # # Post-process action
            # # -------------------------------------------------------------------------- #
            #
            # action_dict = dict(base=base_action, arm=ee_action, gripper=gripper_action)
            # action = env.agent.controller.from_action_dict(action_dict)
            #
            # obs, reward, terminated, truncated, info = env.step(action)
            # print("reward", reward)
            # print("terminated", terminated, "truncated", truncated)
            # print("info", info)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Attempting graceful env shutdown ...")
        env.close()


if __name__ == "__main__":
    main()
