import threading
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Final, Optional, Callable, Generic, TypeVar, Type

from tensordict import TensorDict
from torch import Tensor
import torch

from controller.controllerstep import ControllerStep
from envs import BaseEnvironment
from envs.robotactions import RobotAction
from learnalgorithm.learnalgorithm import LearnAlgorithm
from policy import PolicyBase
from goal.goal import Goal
from utils.human_feedback import HumanFeedback
from utils.logging import log_constructor, logger
from utils.sceneobservation import SceneObservation


@dataclass
class ControllerConfig:
    _CONTROLLER_TYPE: str = field(init=True)
    ACTION_TYPE: str = field(init=True)

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
        self.__last_action: Goal | RobotAction = Goal()
        self._control_variables_lock: threading.Lock = threading.Lock()

        self._goal: Goal = Goal()

        self.__post_step_function: Optional[Callable[[ControllerStep], None]] = None

        self.__step_function: Final[
            Callable[[Goal | RobotAction], tuple[SceneObservation, Tensor, Tensor, TensorDict]]
        ] = (self._environment.step if self._child_controller is None else self._child_controller.set_goal)

        self.__reset_function: Final[Callable] = (
            self._environment.reset if self._child_controller is None else self._child_controller.reset
        )

        if self._child_controller is not None:
            self._child_controller.set_post_step_function(self.child_controller_observation_callback)

    def train(self, mode: bool = True):
        """
        Trains the controller. If the controller has a child controller, it will first train the child, and then
        itself.
        """
        if self._child_controller is not None:
            self._child_controller.train(mode)

        self._policy.train(mode)
        if self._learn_algorithm is not None:
            self._learn_algorithm.train(mode)

    def start(self):
        """
        Starts the controller. If the controller has a child controller, it will first start the child, and then
        itself.
        """
        if self._child_controller is not None:
            self._child_controller.start()

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

        with self._control_variables_lock:
            if self.__last_controller_step.episode_finished:
                return

            action, feedback = self.__sample_action_and_feedback(self._previous_observation)

        if action is None:
            return

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
                reward=self._previous_reward,
                episode_finished=new_episode_finished,
                extra_info=new_extra_info,
            )
            # Only updates if there is no child controller, because the child controller will update the previous
            # at every step of itself.
            if self._child_controller is None:
                self._previous_observation = next_scene_observation

            self._previous_reward = next_reward

            if self._learn_algorithm is not None:
                self._learn_algorithm.step(self.__last_controller_step, feedback)

        if self.__post_step_function is not None:
            self.__post_step_function(self.__last_controller_step)

    def __sample_action_and_feedback(self, scene_observation: SceneObservation) -> (Goal | RobotAction, HumanFeedback):
        action = self._policy(scene_observation)
        if isinstance(action, torch.Tensor):
            action = action.to("cpu")
            # Select the action with the highest probability
            max_index = torch.argmax(action, dim=-1)
            action = torch.zeros_like(action)
            action[:, max_index] = 1
            action = self._action_type.from_label_tensor(action.squeeze(0).detach(), scene_observation)

        # For now one can only compare "Goals" and not "RobotActions"
        if isinstance(action, Goal) and self.__last_action == action:
            return self.__last_action, HumanFeedback.GOOD

        if self._learn_algorithm is not None:
            action, feedback = self._learn_algorithm.get_human_feedback(action, scene_observation)
        else:
            feedback = HumanFeedback.GOOD
        self.__last_action = action
        return action, feedback

    def reset(self) -> SceneObservation:
        self.set_goal(Goal())
        self._policy.episode_finished()
        if self._learn_algorithm is not None:
            self._learn_algorithm.episode_finished()
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
            return_val = self._previous_observation.copy()
        return return_val

    def set_post_step_function(self, post_step_function):
        """
        Sets the function to call after each step.

        Args:
            post_step_function (Callable): The function to call.
        """
        self.__post_step_function = post_step_function

    def save_model(self):
        self._policy.save_to_file()

    def child_controller_observation_callback(self, child_controller_step: ControllerStep):
        with self._control_variables_lock:
            self._previous_observation = child_controller_step.scene_observation
