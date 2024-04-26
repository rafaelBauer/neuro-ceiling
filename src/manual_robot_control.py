import time
from dataclasses import dataclass

import numpy as np

from utils.logging import logger
from omegaconf import OmegaConf, SCMode
# from tqdm.auto import tqdm

from envs import create_environment
from envs.environment import BaseEnvironmentConfig
from utils.argparse import get_config_from_args
from utils.config import ConfigBase


# from utils.keyboard_observer import KeyboardObserver


@dataclass
class Config(ConfigBase):
    env_config: BaseEnvironmentConfig


def create_config_from_args() -> Config:
    extra_args = (
        # {
        #     "name": "--pretraining",
        #     "action": "store_true",
        #     "help": "Whether the data is for pretraining. Used to name the dataset.",
        # },
    )
    args, dict_config = get_config_from_args('Program meant to be used to XXXXX',
                                             data_load=False,
                                             extra_args=extra_args)

    config: Config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )
    return config


def main() -> None:
    np.set_printoptions(suppress=True, precision=3)
    config: Config = create_config_from_args()

    env = create_environment(config.env_config)

    # keyboard_obs = KeyboardObserver()

    env.start()

    time.sleep(5)

    logger.info("Go!")

    try:
        while True:
            action: np.array = np.zeros(7)
            env.step(action)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Attempting graceful env shutdown ...")
        env.close()


if __name__ == "__main__":
    main()
