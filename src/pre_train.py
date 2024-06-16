import os
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
from policy import PolicyBaseConfig, PolicyBase, create_policy
from utils.argparse import get_config_from_args
from utils.config import ConfigBase
from utils.dataset import TrajectoryData
from utils.device import device
from utils.keyboard_observer import KeyboardObserver
from utils.logging import logger


@dataclass
class Config(ConfigBase):
    """
    Configuration class for the collect demonstrations program.
    """

    controllers: list[ControllerConfig]
    policies: list[PolicyBaseConfig]
    environment_config: BaseEnvironmentConfig
    steps: int = 800
    batch_size: int = 16
    feedback_type: str = ""  # This might be overwritten by the command line argument
    dataset_name: str = ""
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
    if args.steps:
        config.steps = args.steps
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
    wandb.init(config=asdict(config), project="neuro-ceiling", mode="online")

    assert len(config.controllers) == len(
        config.policies
    ), "The number of configured controllers and policies must be the same."

    source_path = os.path.join("data/")
    if config.task:
        source_path = os.path.join(source_path, config.task + "/")

    keyboard_obs = KeyboardObserver()
    environment: Final[BaseEnvironment] = create_environment(config.environment_config)

    policies: list[PolicyBase] = []
    controllers: list[ControllerBase] = []

    for policy_config in config.policies:
        policy: PolicyBase = create_policy(policy_config, keyboard_observer=keyboard_obs, environment=environment)
        policies.append(policy)

    # Traverse controllers in reverse order to create the controller hierarchy
    for i, (controller_config) in reversed(list(enumerate(config.controllers))):
        if i == len(config.controllers) - 1:
            controller: ControllerBase = create_controller(controller_config, environment, policies[i])
        else:
            # Always the lower level controller is the child of the higher level controller
            # It will always be in the first position of the list since it was the previously inserted at the index 0
            controller: ControllerBase = create_controller(controller_config, environment, policies[i], controllers[0])
        controllers.insert(0, controller)

    replay_buffer = torch.load(source_path + config.dataset_name)

    logger.info("Pre training starting!")
    try:
        policy = policies[0]
        policy.to(device)
        wandb.watch(policy, log_freq=100)

        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=3e-4,  # learning rate
            weight_decay=3e-6,  # weight decay
        )

        for _ in range(config.steps):
            batch: TrajectoryData = replay_buffer.sample(config.batch_size)
            optimizer.zero_grad()
            batch = batch.to(device)
            losses = []
            for trajectory in batch:
                variance = 0.1 * torch.ones(trajectory.action.size(), dtype=torch.float32)
                variance = variance.to(device)
                loss_function = torch.nn.GaussianNLLLoss()
                out = policy(trajectory.scene_observation)
                # output from model, target, variance
                loss = loss_function(out.squeeze(), trajectory.action, variance)
                loss_with_feedback = loss * trajectory.feedback
                losses.append(loss_with_feedback)
            total_loss = torch.cat(losses).mean()
            total_loss.backward(retain_graph=True)
            optimizer.step()
            policy.reset()
            training_metrics = {"loss": total_loss}
            wandb.log(training_metrics)

        file_name = config.feedback_type + "_policy.pt"
        torch.save(policy.state_dict(), source_path + file_name)
        logger.info("Successfully pre-trained policy for task {}", config.task)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Attempting graceful env shutdown ...")
        controllers[0].stop()
        environment.stop()
        keyboard_obs.stop()


if __name__ == "__main__":
    main()
