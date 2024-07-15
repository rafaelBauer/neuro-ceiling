import time
from dataclasses import dataclass
from typing import Final

import numpy as np
from omegaconf import OmegaConf, SCMode

# from tqdm.auto import tqdm

from controller import create_controller, ControllerBase, ControllerConfig
from envs import BaseEnvironmentConfig, create_environment, BaseEnvironment
from policy import PolicyBaseConfig, PolicyBase, create_policy
from utils.argparse import get_config_from_args
from utils.config import ConfigBase
from utils.keyboard_observer import KeyboardObserver
from utils.logging import logger


@dataclass
class Config(ConfigBase):
    """
    Configuration class for the manual robot control program.
    """

    low_level_controller_config: ControllerConfig
    low_level_policy_config: PolicyBaseConfig
    high_level_controller_config: ControllerConfig
    high_level_policy_config: PolicyBaseConfig
    environment_config: BaseEnvironmentConfig


def create_config_from_args() -> Config:
    """
    Method to create the configuration from the command line arguments.

    Returns:
        Config: The configuration object.
    """
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
    """
    Main method for the manual robot control program.
    """
    np.set_printoptions(suppress=True, precision=3)
    config: Config = create_config_from_args()

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
