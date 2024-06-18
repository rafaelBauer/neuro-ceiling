import os
import time
from dataclasses import dataclass, asdict
from typing import Final

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf, SCMode

from tqdm.auto import tqdm

from controller import create_controller, ControllerBase, ControllerConfig
from controller.controllerstep import ControllerStep
from envs import BaseEnvironmentConfig, create_environment, BaseEnvironment
from learnalgorithm import LearnAlgorithmConfig, LearnAlgorithm, create_learn_algorithm
from policy import PolicyBaseConfig, PolicyBase, create_policy
from utils.argparse import get_config_from_args
from utils.config import ConfigBase
from utils.dataset import TrajectoriesDataset, TrajectoryData
from utils.human_feedback import HumanFeedback
from utils.keyboard_observer import KeyboardObserver
from utils.logging import logger


@dataclass
class Config(ConfigBase):
    """
    Configuration class for the collect demonstrations program.
    """

    controllers: list[ControllerConfig]
    policies: list[PolicyBaseConfig]
    learn_algorithms: list[LearnAlgorithmConfig]
    environment_config: BaseEnvironmentConfig
    episodes: int = 5
    trajectory_size: int = 150
    feedback_type: str = ""  # This might be overwritten by the command line argument
    save_demos: bool = True  # This might be overwritten by the command line argument
    task: str = ""  # This will be overwritten by the command line argument


def create_config_from_args() -> Config:
    """
    Method to create the configuration from the command line arguments.

    Returns:
        Config: The configuration object.
    """
    extra_args = (
        {
            "flag": "-e",
            "name": "--episodes",
            "type": int,
            "help": "Define number of episodes to be executed.",
        },
    )
    args, dict_config = get_config_from_args(
        "Program meant to be used to manually control the robot and eventually collect demonstrations",
        data_load=False,
        extra_args=extra_args,
    )

    config: Config = OmegaConf.to_container(dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE)
    if args.episodes:
        config.episodes = args.episodes
        config.save_demos = True
    if args.task:
        config.task = args.task
    config.feedback_type = args.feedback_type
    return config


def main() -> None:
    """
    Main method for the manual robot control program.
    """
    np.set_printoptions(suppress=True, precision=3)
    config: Config = create_config_from_args()
    wandb.init(config=asdict(config), project="neuro-ceiling", mode="disabled")

    assert len(config.controllers) == len(
        config.policies
    ), "The number of configured controllers and policies must be the same."

    save_path = os.path.join("data/")
    if config.task:
        save_path = os.path.join(save_path, config.task + "/")

    if config.save_demos:
        os.makedirs(save_path, exist_ok=True)

    keyboard_obs = KeyboardObserver()

    environment: Final[BaseEnvironment] = create_environment(config.environment_config)

    policies: list[PolicyBase] = []
    learn_algorithms: list[LearnAlgorithm] = []
    controllers: list[ControllerBase] = []

    for policy_config in config.policies:
        policy: PolicyBase = create_policy(policy_config, keyboard_observer=keyboard_obs, environment=environment)
        policies.append(policy)

    for i, learn_algorithm_config in enumerate(config.learn_algorithms):
        if learn_algorithm_config:
            learn_algorithm: LearnAlgorithm = create_learn_algorithm(learn_algorithm_config, policy=policies[i])
            learn_algorithms.append(learn_algorithm)

    # Traverse controllers in reverse order to create the controller hierarchy
    for i, (controller_config) in reversed(list(enumerate(config.controllers))):
        if i == len(config.controllers) - 1:
            controller: ControllerBase = create_controller(
                controller_config, environment, policies[i], learn_algorithm=learn_algorithms[i]
            )
        else:
            # Always the lower level controller is the child of the higher level controller
            # It will always be in the first position of the list since it was the previously inserted at the index 0
            controller: ControllerBase = create_controller(
                controller_config, environment, policies[i], controllers[0], learn_algorithm=learn_algorithms[i]
            )
        controllers.insert(0, controller)

    replay_buffer = TrajectoriesDataset(config.trajectory_size)
    environment.start()
    time.sleep(5)
    keyboard_obs.start()

    def reset_episode():
        replay_buffer.reset_current_traj()
        controllers[0].reset()

    keyboard_obs.subscribe_callback_to_reset(reset_episode)

    logger.info("Go!")
    try:
        # Have to make as a list to be able to modify it in the post_step_function.
        # In python, immutable objects are passed by value whereas mutable objects are passed by reference.
        episodes_count: list[int] = [0]

        with tqdm(total=config.episodes, desc="Sampling Episodes") as progress_bar:

            def post_step(controller_step: ControllerStep):
                """
                Post step function to save the data to the replay buffer.

                Parameters
                ----------
                controller_step :
                    The arguments from the step function.
                """
                step: TrajectoryData = TrajectoryData(
                    scene_observation=controller_step.scene_observation,
                    action=controller_step.action,
                    feedback=torch.Tensor([HumanFeedback.GOOD]),
                )

                replay_buffer.add(step)

                if controller_step.episode_finished:
                    progress_bar.update(1)
                    replay_buffer.save_current_traj()
                    controllers[0].reset()
                    episodes_count[0] = episodes_count[0] + 1

            controllers[0].set_post_step_function(post_step)
            controllers[0].start()

            while episodes_count[0] < config.episodes:
                # just need to sleep, since there is a thread in the controller doing the stepping and
                # everything else
                time.sleep(1)

        controllers[0].stop()
        environment.stop()
        keyboard_obs.stop()
        file_name = "demos_" + str(config.episodes) + ".dat"
        if config.save_demos:
            torch.save(replay_buffer, save_path + file_name)
        logger.info("Successfully finished to sample {} episodes", episodes_count[0])

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Attempting graceful env shutdown ...")
        controllers[0].stop()
        environment.stop()
        keyboard_obs.stop()


if __name__ == "__main__":
    main()
