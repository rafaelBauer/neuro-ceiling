import os
import time
from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
from omegaconf import OmegaConf, SCMode
from tensordict import TensorDict

from tqdm.auto import tqdm

from controller import create_controller, ControllerBase, ControllerConfig
from controller.controllerstep import ControllerStep
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

    controllers: list[ControllerConfig]
    policies: list[PolicyBaseConfig]
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
    )
    args, dict_config = get_config_from_args(
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

    assert len(config.controllers) == len(
        config.policies
    ), "The number of configured controllers and policies must be the same."

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

    policies: list[PolicyBase] = []
    controllers: list[ControllerBase] = []

    for policy_config in config.policies:
        policy: PolicyBase = create_policy(
            policy_config, keyboard_observer=keyboard_obs, environment=environment, scene=scene
        )
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
