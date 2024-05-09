import time
from dataclasses import dataclass

import numpy as np
from omegaconf import OmegaConf, SCMode

# from tqdm.auto import tqdm

from agent import create_agent
from agent.agent import AgentConfig
from utils.argparse import get_config_from_args
from utils.config import ConfigBase
from utils.keyboard_observer import KeyboardObserver
from utils.logging import logger


# from utils.keyboard_observer import KeyboardObserver


@dataclass
class Config(ConfigBase):
    agent_config: AgentConfig


def create_config_from_args() -> Config:
    extra_args = (
        # {
        #     "name": "--pretraining",
        #     "action": "store_true",
        #     "help": "Whether the data is for pretraining. Used to name the dataset.",
        # },
    )
    _, dict_config = get_config_from_args(
        "Program meant to be used to XXXXX", data_load=False, extra_args=extra_args
    )

    config: Config = OmegaConf.to_container(dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE)
    return config


def main() -> None:
    np.set_printoptions(suppress=True, precision=3)
    config: Config = create_config_from_args()

    agent = create_agent(config.agent_config)

    keyboard_obs = KeyboardObserver()

    logger.info("Go!")

    keyboard_obs.start()
    agent.start()

    try:
        while True:
            # just need to sleep, since there is a thread in the agent doing the stepping and
            # everything else
            time.sleep(0.004)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Attempting graceful env shutdown ...")
        keyboard_obs.stop()
        agent.stop()


if __name__ == "__main__":
    main()
