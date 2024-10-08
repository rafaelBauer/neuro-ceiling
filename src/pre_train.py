import os
import time
from dataclasses import dataclass, asdict
from typing import Final

import numpy as np
import wandb
from omegaconf import OmegaConf, SCMode

from tqdm.auto import tqdm

from controller import create_controller, ControllerBase, ControllerConfig
from controller.controllerstep import ControllerStep
from envs import BaseEnvironmentConfig, create_environment, BaseEnvironment
from envs.maniskill import ManiSkillEnvironmentConfig
from learnalgorithm import LearnAlgorithmConfig, create_learn_algorithm, LearnAlgorithm
from policy import PolicyBaseConfig, PolicyBase, create_policy
from utils.argparse import get_config_from_args
from utils.config import ConfigBase
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
    episodes: int = 100
    batch_size: int = 16
    train: bool = True
    task: str = ""  # This will be overwritten by the command line argument


def create_config_from_args() -> Config:
    """
    Method to create the configuration from the command line arguments.

    Returns:
        Config: The configuration object.
    """
    extra_args = (
        {
            "flag": "-s",
            "name": "--steps",
            "type": int,
            "help": "Define number of training steps to take.",
        },
    )
    args, dict_config = get_config_from_args(
        "Program meant to pretrain a model based on demonstrations", data_load=False, extra_args=extra_args
    )

    config: Config = OmegaConf.to_container(dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE)
    if args.task:
        config.task = args.task
    if args.model_file:
        config.policies[0].from_file = args.model_file
    return config


def main() -> None:
    """
    Main method for the manual robot control program.
    """
    np.set_printoptions(suppress=True, precision=3)
    config: Config = create_config_from_args()

    if config.train:
        job_type = "train"
    else:
        job_type = "evaluation"

    wandb.init(config=asdict(config), project=config.task, mode="online", job_type=job_type)

    assert len(config.controllers) == len(
        config.policies
    ), "The number of configured controllers and policies must be the same."

    source_path = os.path.join("data/")
    if config.task:
        source_path = os.path.join(source_path, config.task + "/")

    if isinstance(config.environment_config, ManiSkillEnvironmentConfig) and config.environment_config.headless:
        keyboard_obs = None
    else:
        from utils.keyboard_observer import KeyboardObserver

        keyboard_obs = KeyboardObserver()

    environment: Final[BaseEnvironment] = create_environment(config.environment_config)

    policies: list[PolicyBase] = []
    learn_algorithms: list[LearnAlgorithm] = []
    controllers: list[ControllerBase] = []

    for policy_config in config.policies:
        if policy_config.from_file:
            policy_config.from_file = os.path.join(source_path, policy_config.from_file)

        if policy_config.save_to_file:
            filename, file_extension = os.path.splitext(policy_config.save_to_file)
            policy_config.save_to_file = f"{filename}_{str(config.episodes)}{file_extension}"
            policy_config.save_to_file = os.path.join(source_path, policy_config.save_to_file)

        policy: PolicyBase = create_policy(policy_config, keyboard_observer=keyboard_obs, environment=environment)
        policy.load_from_file()
        policies.append(policy)

    for i, learn_algorithm_config in enumerate(config.learn_algorithms):
        if learn_algorithm_config:
            if learn_algorithm_config.load_dataset:
                learn_algorithm_config.load_dataset = source_path + learn_algorithm_config.load_dataset
            if learn_algorithm_config.save_dataset:
                filename, file_extension = os.path.splitext(learn_algorithm_config.save_dataset)
                learn_algorithm_config.save_dataset = (
                    source_path + filename + "_" + str(config.episodes) + file_extension
                )

            learn_algorithm: LearnAlgorithm = create_learn_algorithm(
                learn_algorithm_config, policy=policies[i], keyboard_observer=keyboard_obs
            )
            if learn_algorithm:
                learn_algorithm.load_dataset()
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

    def reset_episode():
        controllers[0].reset()

    if keyboard_obs is not None:
        keyboard_obs.subscribe_callback_to_reset(reset_episode)

    try:
        if config.train:
            logger.info("Training starting!")
            controllers[0].train()
        else:
            wandb.run.tags = ["Evaluation"]

        if config.episodes > 0:
            logger.info("Sampling episodes starting!")

            environment.start()
            time.sleep(5)
            if keyboard_obs is not None:
                keyboard_obs.start()

            # Have to make as a list to be able to modify it in the post_step_function.
            # In python, immutable objects are passed by value whereas mutable objects are passed by reference.
            episodes_count: list[int] = [0]

            with tqdm(total=config.episodes, desc="Sampling Episodes", position=0) as progress_bar:

                def post_step(controller_step: ControllerStep):
                    if controller_step.episode_finished:
                        progress_bar.update(1)
                        controllers[0].reset()
                        episodes_count[0] = episodes_count[0] + 1

                controllers[0].set_post_step_function(post_step)
                controllers[0].start()

                while episodes_count[0] < config.episodes:
                    # just need to sleep, since there is a thread in the controller doing the stepping and
                    # everything else
                    time.sleep(1)

            if config.train:
                controllers[0].train(False)

            controllers[0].stop()
            environment.stop()
            if keyboard_obs is not None:
                keyboard_obs.stop()

        if config.train:
            controllers[0].publish_model(config.train)
            logger.info("Successfully trained policy for task {}", config.task)

        if config.episodes > 0 and learn_algorithms[0] is not None:
            controllers[0].publish_dataset()
            wandb.run.summary["learn_algorithm"] = config.learn_algorithms[0].name

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Attempting graceful env shutdown ...")
        controllers[0].stop()
        environment.stop()
        if keyboard_obs is not None:
            keyboard_obs.stop()


if __name__ == "__main__":
    main()
