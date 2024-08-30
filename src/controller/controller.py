import threading
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Final, Optional, Callable, Generic, TypeVar, Type

import wandb
from tensordict import TensorDict
from torch import Tensor
import torch

from controller.controllerstep import ControllerStep
from envs import BaseEnvironment
from envs.robotactions import RobotAction
from learnalgorithm.learnalgorithm import LearnAlgorithm
from policy import PolicyBase
from goal.goal import Goal
from utils.device import device
from utils.human_feedback import HumanFeedback
from utils.logging import log_constructor, logger
from utils.metricslogger import MetricsLogger, EpisodeMetrics
from utils.sceneobservation import SceneObservation


@dataclass
class ControllerConfig:
    _CONTROLLER_TYPE: str = field(init=True)
    ACTION_TYPE: str = field(init=True)
    initial_goal: list = field(init=True)
    max_steps: int = field(init=True, default=0)
    log_metrics: bool = field(init=True, default=False)

    @property
    def controller_type(self) -> str:
        return self._CONTROLLER_TYPE


class ControllerBase:
    """
    The ControllerBase class is the base class for all controllers.

    Attributes:
        __CONFIG (ControllerConfig): The configuration for the controller.
        _environment (BaseEnvironment): The environment in which the controller operates.
        _policy (PolicyBase): The policy used by the controller.
        _child_controller (Optional[ControllerBase]): The child controller, if any.
        _previous_observation (SceneObservation): The previous observation made by the controller.
        _previous_reward (Tensor): The previous reward received by the controller.
        __last_controller_step (ControllerStep): The last step taken by the controller.
        _control_variables_lock (threading.Lock): A lock for controlling access to variables.
        _goal (Goal): The current goal of the controller.
        __post_step_function (Optional[Callable[[ControllerStep], None]]): The function to call after each step.
        __step_function (Callable): The function to call for each step.
                                    It is either the environment's step function or the child controller's set_goal function.
    """

    # @log_constructor
    def __init__(
        self,
        config: ControllerConfig,
        environment: BaseEnvironment,
        policy: PolicyBase,
        action_type: Type = Goal,
        child_controller: Optional["ControllerBase"] = None,
        learn_algorithm: Optional[LearnAlgorithm] = None,
    ):
        """
        Initializes the ControllerBase class.

        Args:
            config (ControllerConfig): The configuration for the controller.
            environment (BaseEnvironment): The environment in which the controller operates.
            policy (PolicyBase): The policy used by the controller.
            child_controller (Optional[ControllerBase]): The child controller, if any.
        """
        # Type of the action executed by this controller
        self._action_type = action_type

        self.__CONFIG: Final[ControllerConfig] = config
        self._environment: Final[BaseEnvironment] = environment
        self._policy: Final[PolicyBase] = policy
        self._learn_algorithm: Final[LearnAlgorithm] = learn_algorithm
        self._child_controller: Optional[ControllerBase] = child_controller

        self._policy.to(device)
        # Control variables for learning
        self._previous_observation: SceneObservation = SceneObservation(
            camera_observation={},
            proprioceptive_obs=torch.tensor([]),
            end_effector_pose=torch.tensor([]),
            objects={},
            spots={},
        )
        self._previous_reward: Tensor = torch.tensor(0.0)
        self.__last_controller_step: ControllerStep = ControllerStep(
            action=Tensor(),
            scene_observation=self._previous_observation,
            reward=self._previous_reward,
            episode_finished=False,
            extra_info={},
        )
        self.__last_action: Goal | RobotAction = Goal(input_tensor=Tensor(config.initial_goal))
        self._control_variables_lock: threading.Lock = threading.Lock()

        self._goal: Goal = Goal()

        self.__post_step_function: Optional[Callable[[ControllerStep], None]] = None

        self.__step_function: Final[
            Callable[[Goal | RobotAction], tuple[SceneObservation, Tensor, Tensor, TensorDict]]
        ] = (self._environment.step if self._child_controller is None else self._child_controller.set_goal)

        self.__reset_function: Final[Callable[[], SceneObservation]] = (
            self._environment.reset if self._child_controller is None else self._child_controller.reset
        )

        if self._child_controller is not None:
            self._child_controller.set_post_step_function(self.child_controller_observation_callback)

        if self.__CONFIG.log_metrics:
            self._metrics_logger = MetricsLogger()

        else:
            self._episode_metrics = None
            self._metrics_logger = None

        if self._learn_algorithm is not None:
            self._learn_algorithm.set_metrics_logger(self._metrics_logger)
            self._learn_algorithm.set_feedback_update_callback(self.feedback_update_callback)

        self.__training_mode: bool = False

        self.__step_number: int = 0

    def train(self, mode: bool = True):
        """
        Trains the controller. If the controller has a child controller, it will first train the child, and then
        itself.
        """
        if self._child_controller is not None:
            self._child_controller.train(mode)

        self.__training_mode = mode

        if mode:
            self._policy.train(mode)
            if self._learn_algorithm is not None:
                self._learn_algorithm.train(mode)
        else:
            if self._learn_algorithm is not None:
                self._learn_algorithm.train(mode)
            self._policy.train(mode)

    def eval(self, steps_limit: int = 0):
        self.train(False)

    def start(self):
        """
        Starts the controller. If the controller has a child controller, it will first start the child, and then
        itself.
        """
        if self._child_controller is not None:
            self._child_controller.start()

        self._previous_observation = self.__reset_function()
        if self.__CONFIG.log_metrics:
            self._episode_metrics = EpisodeMetrics(0, self._previous_observation.objects)

        self._specific_start()

    @abstractmethod
    def _specific_start(self):
        """
        Meant to be overridden by subclasses, this method allows a specific subclass to perform the necessary steps, so
        it gets started
        """

    def stop(self):
        """
        Stops the controller. If the controller has a child controller, it will first stop the child, and then the
        itself.
        """
        if self._child_controller is not None:
            self._child_controller.stop()

        self._specific_stop()
        if self.__CONFIG.log_metrics:
            self._metrics_logger.log_session()

    @abstractmethod
    def _specific_stop(self):
        """
        Meant to be overridden by subclasses, this method allows a specific subclass to perform the necessary steps, so
        it gets stopped
        """

    def set_goal(self, goal: Goal) -> tuple[SceneObservation, Tensor, Tensor, TensorDict]:
        """
        Analogous to the `step` method from an environment, this method sets the goal for the controller, which
        will "notify" its policy about the objective of the task.

        Args:
            goal (Goal): The goal to set.

        Returns:
            tuple[SceneObservation, Tensor, Tensor, TensorDict]: The last controller step.
        """
        if not (self._goal == goal):
            self._goal = goal
            logger.debug("Setting goal {} to policy", self._goal)
            self._policy.goal_to_be_achieved(self._goal)
        with self._control_variables_lock:
            return (
                self.__last_controller_step.scene_observation,
                self.__last_controller_step.reward,
                self.__last_controller_step.episode_finished,
                self.__last_controller_step.extra_info,
            )

    def _step(self):
        """
        Performs a step in the controller. This method is responsible for executing the action and updating the state
        of the controller.
        """
        if 0 < self.__CONFIG.max_steps < self.__step_number:
            self.__last_controller_step.episode_finished = True
            if self.__post_step_function is not None:
                self.__post_step_function(self.__last_controller_step)
            return

        with self._control_variables_lock:
            observation_copy = self._previous_observation.copy()
        action, feedback = self.__sample_action_and_feedback(observation_copy)

        assert isinstance(
            action, Goal | RobotAction
        ), f"Action should be of type RobotAction or Goal, but got {type(action)}"

        # Perform a step either into the environment or into the child controller
        (next_scene_observation, next_reward, new_episode_finished, new_extra_info) = self.__step_function(action)

        with self._control_variables_lock:
            # To fit into the general RL framework, the controller step consists of current action taken and the
            # states and rewards that lead to the chosen action (previous state and rewards).
            # (S_{t-1}, R_{t-1}, a_t)
            self.__last_action = action
            self.__last_controller_step = ControllerStep(
                action=action.to_tensor(),
                scene_observation=self._previous_observation,
                reward=next_reward,
                episode_finished=new_episode_finished,
                extra_info=new_extra_info,
            )

            if self.__CONFIG.log_metrics:
                self._episode_metrics.log_step(self.__last_controller_step.reward, feedback)
            # Only updates if there is no child controller, because the child controller will update the previous
            # at every step of itself.
            if self._child_controller is None:
                self._previous_observation = next_scene_observation
                self._previous_reward = next_reward

            if self._learn_algorithm is not None:
                self._learn_algorithm.save_current_step(self.__last_controller_step, feedback)

        if self.__post_step_function is not None:
            self.__post_step_function(self.__last_controller_step)

        self.__step_number += 1

    def __sample_action_and_feedback(self, scene_observation: SceneObservation) -> (Goal | RobotAction, HumanFeedback):
        action: Tensor = self._policy(scene_observation)
        action = action.squeeze(0)
        action: RobotAction | Goal = self._action_type.from_tensor(action.squeeze(0).detach(), scene_observation)

        if self._learn_algorithm is not None:
            action, feedback = self._learn_algorithm.get_human_feedback(action, scene_observation)
        else:
            feedback = HumanFeedback.GOOD

        # For now one can only compare "Goals" and not "RobotActions"
        if action is not None and self.__last_action != action and self.__last_action.replaceable(action):
            self.__last_action = action

        return self.__last_action, feedback

    def reset(self) -> SceneObservation:
        self.__step_number = 0

        if self.__CONFIG.log_metrics:
            self._metrics_logger.log_episode(self._episode_metrics)

        self.set_goal(Goal(input_tensor=Tensor(self.__CONFIG.initial_goal)))
        self._policy.episode_finished()
        if self._learn_algorithm is not None:
            self._learn_algorithm.episode_finished()
            self._learn_algorithm.reset()
        elif self.__CONFIG.log_metrics:
            wandb.log(self._metrics_logger.pop())

        with self._control_variables_lock:
            self._previous_reward = torch.tensor(0.0)
            self._previous_observation = self.__reset_function()
            self.__last_controller_step: ControllerStep = ControllerStep(
                action=Tensor(),
                scene_observation=self._previous_observation,
                reward=self._previous_reward,
                episode_finished=False,
                extra_info={},
            )
            return_val: SceneObservation = self._previous_observation.copy()
        # Log current episode metrics and create new episode metrics object
        if self.__CONFIG.log_metrics:
            self._episode_metrics = EpisodeMetrics(self._episode_metrics.episode_number + 1, return_val.objects)

        return return_val

    def set_post_step_function(self, post_step_function):
        """
        Sets the function to call after each step.

        Args:
            post_step_function (Callable): The function to call.
        """
        self.__post_step_function = post_step_function

    def publish_model(self, trained_model: bool = False):
        self._policy.publish_model(trained_model)

    def publish_dataset(self):
        if self._learn_algorithm is not None:
            self._learn_algorithm.publish_dataset()

    def child_controller_observation_callback(self, child_controller_step: ControllerStep):
        with self._control_variables_lock:
            self._previous_observation = child_controller_step.scene_observation
            self._previous_reward = child_controller_step.reward

    def feedback_update_callback(self, feedback: HumanFeedback):
        if self.__CONFIG.log_metrics:
            self._episode_metrics.update_current_step_feedback(feedback)
