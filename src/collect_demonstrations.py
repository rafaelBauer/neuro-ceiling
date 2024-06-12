import os
import time
from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
from omegaconf import OmegaConf, SCMode
from tensordict import TensorDict
from torch import Tensor

from tqdm.auto import tqdm

from controller import create_controller, ControllerBase, ControllerConfig
from envs import BaseEnvironmentConfig, create_environment, BaseEnvironment
from envs.object import Object, Spot
from envs.scene import Scene
from goal.goal import Goal
from policy import PolicyBaseConfig, PolicyBase, create_policy
from utils.argparse import get_config_from_args
from utils.config import ConfigBase
from utils.dataset import TrajectoriesDataset, TrajectoryData
from utils.human_feedback import HumanFeedback
from utils.keyboard_observer import KeyboardObserver
from utils.logging import logger
from utils.pose import Pose
from utils.sceneobservation import SceneObservation


@dataclass
class Config(ConfigBase):
    """
    Configuration class for the collect demonstrations program.
    """

    low_level_controller_config: ControllerConfig
    low_level_policy_config: PolicyBaseConfig
    high_level_controller_config: ControllerConfig
    high_level_policy_config: PolicyBaseConfig
    environment_config: BaseEnvironmentConfig
    episodes: int = 5
    trajectory_size: int = 150
    save_demos: bool = True


def create_config_from_args() -> Config:
    """
    Method to create the configuration from the command line arguments.

    Returns:
        Config: The configuration object.
    """
    extra_args = (
        {
            "name": "--pretraining",
            "action": "store_true",
            "help": "Whether the data is for pretraining. Used to name the dataset.",
        },
        {
            "name": "--episodes",
            "action": "store_true",
            "default": 10,
            "help": "How many episodes to collect.",
        },
    )
    args, dict_config = get_config_from_args(
        "Program meant to be used to manually control the robot", data_load=False, extra_args=extra_args
    )

    config: Config = OmegaConf.to_container(dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE)
    config.episodes = args.episodes
    return config


def post_step_function(
    args: tuple[SceneObservation, float, bool, dict],
    action: Tensor,
    replay_buffer: TrajectoriesDataset,
    episodes_count,
    progress_bar,
    keyboard_obs
) -> None:
    """
    Post step function to save the data to the replay buffer.

    Parameters
    ----------
    args : tuple[dict, float, bool, dict]
        The arguments from the step function.
    replay_buffer : TrajectoriesDataset
        The replay buffer to save the data to.
    """
    observation, reward, done, info = args
    step: TrajectoryData = TrajectoryData()
    step.sceneObservation = observation
    step.feedback = torch.Tensor([HumanFeedback.GOOD])
    step.action = action
    # step.object_poses = TensorDict()
    # step.spots = TensorDict()

    replay_buffer.add(step)

    if done is True:
        episodes_count += 1
        progress_bar.update(1)


def main() -> None:
    """
    Main method for the manual robot control program.
    """
    np.set_printoptions(suppress=True, precision=3)
    config: Config = create_config_from_args()

    save_path = os.path.join("data/")
    os.makedirs(save_path, exist_ok=True)

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

    replay_buffer = TrajectoriesDataset(config.trajectory_size)
    environment.start()
    time.sleep(5)
    keyboard_obs.start()

    logger.info("Go!")
    try:
        episodes_count = 0

        with tqdm(total=config.episodes, desc="Sampling Episodes") as progress_bar:
            high_level_controller.set_post_step_function(
                lambda args, action: post_step_function(args, action, replay_buffer, episodes_count, progress_bar, keyboard_obs)
            )
            high_level_controller.start()
            while episodes_count < config.episodes:
                # just need to sleep, since there is a thread in the controller doing the stepping and
                # everything else
                time.sleep(2)

        high_level_controller.stop()
        file_name = "demos_" + str(config.episodes) + ".dat"
        if config.save_demos:
            torch.save(replay_buffer, save_path + file_name)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Attempting graceful env shutdown ...")
        high_level_controller.stop()
        environment.stop()
        keyboard_obs.stop()


if __name__ == "__main__":
    main()
