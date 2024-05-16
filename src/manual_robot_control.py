import time
from dataclasses import dataclass
from typing import Final

import numpy as np
from omegaconf import OmegaConf, SCMode

# from tqdm.auto import tqdm

from agent import create_agent, AgentBase, AgentConfig
from envs import BaseEnvironmentConfig, create_environment, BaseEnvironment
from policy import PolicyBaseConfig, PolicyBase, create_policy
from utils.argparse import get_config_from_args
from utils.config import ConfigBase
from utils.keyboard_observer import KeyboardObserver
from utils.logging import logger


@dataclass
class Config(ConfigBase):
    agent_config: AgentConfig
    policy_config: PolicyBaseConfig
    environment_config: BaseEnvironmentConfig


def create_config_from_args() -> Config:
    extra_args = (
        # {
        #     "name": "--pretraining",
        #     "action": "store_true",
        #     "help": "Whether the data is for pretraining. Used to name the dataset.",
        # },
    )
    _, dict_config = get_config_from_args("Program meant to be used to XXXXX", data_load=False, extra_args=extra_args)

    config: Config = OmegaConf.to_container(dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE)
    return config


def main() -> None:
    np.set_printoptions(suppress=True, precision=3)
    config: Config = create_config_from_args()

    environment: Final[BaseEnvironment] = create_environment(config.environment_config)
    policy: Final[PolicyBase] = create_policy(config.policy_config)
    agent: Final[AgentBase] = create_agent(config.agent_config, environment, policy)

    # keyboard_obs = KeyboardObserver()

    environment.start()
    time.sleep(5)

    logger.info("Go!")

    # keyboard_obs.start()
    agent.start()

    try:
        while True:
            # just need to sleep, since there is a thread in the agent doing the stepping and
            # everything else
            time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Attempting graceful env shutdown ...")
        # keyboard_obs.stop()
        agent.stop()


if __name__ == "__main__":
    main()
