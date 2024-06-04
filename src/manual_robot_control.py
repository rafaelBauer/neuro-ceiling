import time
from dataclasses import dataclass
from typing import Final

import numpy as np
from omegaconf import OmegaConf, SCMode

# from tqdm.auto import tqdm

from controller import create_controller, ControllerBase, ControllerConfig
from envs import BaseEnvironmentConfig, create_environment, BaseEnvironment
from envs.object import Object, Spot
from envs.scene import Scene
from policy import PolicyBaseConfig, PolicyBase, create_policy
from goal.movetoposition import MoveObjectToPosition
from utils.argparse import get_config_from_args
from utils.config import ConfigBase
from utils.keyboard_observer import KeyboardObserver
from utils.logging import logger
from utils.pose import Pose


@dataclass
class Config(ConfigBase):
    low_level_controller_config: ControllerConfig
    low_level_policy_config: PolicyBaseConfig
    high_level_controller_config: ControllerConfig
    high_level_policy_config: PolicyBaseConfig
    environment_config: BaseEnvironmentConfig


def create_config_from_args() -> Config:
    extra_args = (
        # {
        #     "name": "--pretraining",
        #     "action": "store_true",
        #     "help": "Whether the data is for pretraining. Used to name the dataset.",
        # },
    )
    _, dict_config = get_config_from_args(
        "Program meant to be used to manually control the robot", data_load=False, extra_args=extra_args
    )

    config: Config = OmegaConf.to_container(dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE)
    return config


def main() -> None:
    np.set_printoptions(suppress=True, precision=3)
    config: Config = create_config_from_args()

    cube_a: Object = Object(Pose(p=[0.615, -0.2, 0.02], q=[0, 1, 0, 0]))
    cube_b: Object = Object(Pose(p=[0.615, 0.0, 0.02], q=[0, 1, 0, 0]))
    cube_c: Object = Object(Pose(p=[0.615, 0.2, 0.02], q=[0, 1, 0, 0]))

    spot_a: Spot = Spot(object=cube_a)
    spot_b: Spot = Spot(object=cube_b)
    spot_c: Spot = Spot(object=cube_c)

    scene: Scene = Scene([cube_a, cube_b, cube_c], [spot_a, spot_b, spot_c])

    keyboard_obs = KeyboardObserver()

    environment: Final[BaseEnvironment] = create_environment(config.environment_config)
    low_level_policy: Final[PolicyBase] = create_policy(
        config.low_level_policy_config, keyboard_observer=keyboard_obs, environment=environment
    )
    low_level_controller: Final[ControllerBase] = create_controller(
        config.low_level_controller_config, environment, low_level_policy
    )

    high_level_policy: Final[PolicyBase] = create_policy(
        config.high_level_policy_config, keyboard_observer=keyboard_obs, environment=environment, scene=scene
    )
    high_level_controller: Final[ControllerBase] = create_controller(
        config.high_level_controller_config, environment, high_level_policy, low_level_controller
    )

    environment.start()
    time.sleep(5)

    logger.info("Go!")

    try:
        keyboard_obs.start()
        high_level_controller.start()

        time.sleep(5)

        # # MoveObjectToPosition()
        # object_pose = Pose(p=[0.615, 0, 0.02], q=[0, 1, 0, 0])
        # target_pose = Pose(p=[0.615, 0.2, 0.06], q=[0, 1, 0, 0])
        # move_object_task = MoveObjectToPosition(object_pose, target_pose)
        # low_level_controller.set_goal(move_object_task)
        # time.sleep(15)
        # object_pose2 = Pose(p=[0.615, -0.2, 0.02], q=[0, 1, 0, 0])
        # target_pose2 = Pose(p=[0.615, 0.2, 0.1], q=[0, 1, 0, 0])
        # move_object_task = MoveObjectToPosition(object_pose2, target_pose2)
        # low_level_controller.set_goal(move_object_task)

        while True:
            # just need to sleep, since there is a thread in the controller doing the stepping and
            # everything else
            time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Attempting graceful env shutdown ...")
        high_level_controller.stop()
        environment.stop()
        keyboard_obs.stop()


if __name__ == "__main__":
    main()
