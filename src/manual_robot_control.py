import time
from dataclasses import dataclass
from typing import Final

import numpy as np
from omegaconf import OmegaConf, SCMode

# from tqdm.auto import tqdm

from agent import create_agent, AgentBase, AgentConfig
from envs import BaseEnvironmentConfig, create_environment, BaseEnvironment
from policy import PolicyBaseConfig, PolicyBase, create_policy
from task.movetoposition import MoveObjectToPosition
from utils.argparse import get_config_from_args
from utils.config import ConfigBase
from utils.keyboard_observer import KeyboardObserver
from utils.logging import logger
from utils.pose import Pose


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

    keyboard_obs = KeyboardObserver()

    environment: Final[BaseEnvironment] = create_environment(config.environment_config)
    policy: Final[PolicyBase] = create_policy(
        config.policy_config, keyboard_observer=keyboard_obs, environment=environment
    )
    agent: Final[AgentBase] = create_agent(config.agent_config, environment, policy)

    environment.start()
    time.sleep(5)

    logger.info("Go!")

    try:
        keyboard_obs.start()
        agent.start()

        # time.sleep(5)
        #
        # # MoveObjectToPosition()
        # object_pose = Pose(p=[0.615, 0, 0.02], q=[0, 1, 0, 0])
        # target_pose = Pose(p=[0.615, 0.2, 0.06], q=[0, 1, 0, 0])
        # move_object_task = MoveObjectToPosition(object_pose, target_pose)
        # agent.execute_task(move_object_task)
        # time.sleep(15)
        # object_pose2 = Pose(p=[0.615, -0.2, 0.02], q=[0, 1, 0, 0])
        # target_pose2 = Pose(p=[0.615, 0.2, 0.1], q=[0, 1, 0, 0])
        # move_object_task = MoveObjectToPosition(object_pose2, target_pose2)
        # agent.execute_task(move_object_task)

        while True:
            # just need to sleep, since there is a thread in the agent doing the stepping and
            # everything else
            time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Attempting graceful env shutdown ...")
        agent.stop()
        keyboard_obs.stop()


if __name__ == "__main__":
    main()
