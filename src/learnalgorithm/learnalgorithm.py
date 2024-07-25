import os.path
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Final, TypeVar, Type, Optional, Callable

import torch
import wandb
from torch import nn
from torch.utils.data import Sampler, DataLoader

from controller.controllerstep import ControllerStep
from envs.robotactions import RobotAction
from goal.goal import Goal
from policy import PolicyBase
from utils.dataset import TrajectoriesDataset, TrajectoryData
from utils.device import device
from utils.human_feedback import HumanFeedback
from utils.logging import logger
from utils.metricslogger import MetricsLogger
from utils.sceneobservation import SceneObservation


@dataclass
class LearnAlgorithmConfig:
    _ALGO_TYPE: str = field(init=False)
    batch_size: int = field(init=True)
    learning_rate: float = field(init=True)
    weight_decay: float = field(init=True)
    steps_per_episode: int = field(init=True)
    load_dataset: str = field(init=True)
    save_dataset: str = field(init=True)

    @property
    def algo_type(self) -> str:
        return self._ALGO_TYPE


@dataclass
class NoLearnAlgorithmConfig(LearnAlgorithmConfig):
    """
    This is a dataclass that represents the configuration when there should not be any learn algorithm.
    """

    _ALGO_TYPE: str = field(init=False, default="NoLearnAlgorithm")
    batch_size: int = field(init=False, default=0)
    learning_rate: float = field(init=False, default=0)
    weight_decay: float = field(init=False, default=0)
    steps_per_episode: int = field(init=False, default=0)
    load_dataset: str = field(init=True, default="")
    save_dataset: str = field(init=True, default="")


class LearnAlgorithm:
    def __init__(
        self,
        config: LearnAlgorithmConfig,
        policy: PolicyBase,
        sampler: Type[Sampler],
        dataloader: Type[DataLoader],
        loss_function: nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        self._policy = policy
        self._CONFIG: Final[LearnAlgorithmConfig] = config

        self.__sampler_type = sampler
        self.__dataloader_type = dataloader

        self._replay_buffer: TrajectoriesDataset = TrajectoriesDataset(config.steps_per_episode)
        self._sampler: Optional[Sampler] = None
        self._dataloader: Optional[DataLoader] = None

        # Which loss function to use for the algorithm
        self._loss_function = loss_function

        # Optimizer
        self._optimizer = optimizer

        self._metrics_logger = MetricsLogger()

        self._feedback_update_callback: Optional[Callable[[HumanFeedback], None]] = None

        wandb.watch(policy, log_freq=100)

    def load_dataset(self):
        """
        This method is responsible for loading a dataset from a file into the replay buffer.

        It first checks if the load_dataset attribute of the configuration object is set. If it is not set, it returns immediately.
        Then it loads the dataset from the file specified by the load_dataset attribute of the configuration object into the replay buffer.
        It creates a sampler for the replay buffer.
        It creates a dataloader for the replay buffer with the created sampler, the batch size specified in the configuration object, and a collate function that returns the input as is.
        """
        if self._CONFIG.load_dataset:
            try:
                file_name_and_extension = os.path.basename(self._CONFIG.load_dataset)
                task_name = os.path.dirname(self._CONFIG.load_dataset).split("/")[-1]
                artifact = wandb.run.use_artifact(f"{task_name}/{os.path.splitext(file_name_and_extension)[0]}:latest")
                dataset_file = artifact.download()
            except wandb.CommError as exception:
                logger.info("Could not download artifact from wandb: {}", exception)
                logger.info("Using dataset from local filesystem: {}", self._CONFIG.load_dataset)
                dataset_file = self._CONFIG.load_dataset

            logger.info("Loading dataset from file: {}", dataset_file)
            self._replay_buffer: TrajectoriesDataset = torch.load(dataset_file)
            self._sampler = self.__sampler_type(self._replay_buffer)
            self._dataloader = self.__dataloader_type(
                dataset=self._replay_buffer,
                sampler=self._sampler,
                batch_size=self._CONFIG.batch_size,
                collate_fn=lambda x: x,
            )

    def save_dataset(self):
        """
        This method is responsible for saving the current state of the replay buffer to a file.

        It first checks if the save_dataset attribute of the configuration object is set. If it is not set, it returns immediately.
        Then it saves the replay buffer to the file specified by the save_dataset attribute of the configuration object.
        Finally, it logs a message indicating that the dataset was successfully saved.
        """
        if self._CONFIG.save_dataset:
            torch.save(self._replay_buffer, self._CONFIG.save_dataset)
            logger.info("Successfully saved dataset {}", self._CONFIG.save_dataset)

    def publish_dataset(self):
        if self._CONFIG.save_dataset:
            self.save_dataset()
            file_name_and_extension = os.path.basename(self._CONFIG.save_dataset)
            task_name = os.path.dirname(self._CONFIG.save_dataset).split("/")[-1]
            raw_data = wandb.Artifact(
                f"{os.path.splitext(file_name_and_extension)[0]}",
                type="dataset",
                description=f"{str(len(self._replay_buffer))} trajectories from task {task_name}",
                metadata={"task": task_name, "source": "TrajectoriesDataset", "sizes": len(self._replay_buffer)},
            )
            with raw_data.new_file(file_name_and_extension, mode="wb") as file:
                torch.save(self._replay_buffer, file)
            wandb.log_artifact(raw_data)

    def set_metrics_logger(self, metrics_logger: MetricsLogger):
        """
        This method is responsible for setting the metrics logger of the learn algorithm.

        Args:
            metrics_logger (MetricsLogger): The metrics logger to set.
        """
        self._metrics_logger = metrics_logger

    def set_feedback_update_callback(self, feedback_update_callback: Callable[[HumanFeedback], None]):
        """
        This method is responsible for setting the feedback update callback of the learn algorithm.

        Args:
            feedback_update_callback: The feedback update callback to set.
        """
        self._feedback_update_callback = feedback_update_callback

    @abstractmethod
    def train(self, mode: bool = True):
        """
        This is an abstract method that is responsible for training the algorithm.

        Args:
            mode (bool): A flag indicating whether the algorithm should be trained. Defaults to True.

        It needs to be implemented in the subclass.
        """
        pass

    def get_human_feedback(
        self, next_action: Goal | RobotAction, scene_observation: SceneObservation
    ) -> (Goal | RobotAction, HumanFeedback):
        """
        This method is responsible for getting the teachers' feedback for the next action.

        It is meant to be overwritten by the subclass if the algorithm needs to get feedback from the teacher.
        Args:
            next_action (Goal | RobotAction): The next action that the robot will take.
            scene_observation (SceneObservation): The current scene observation.

        Returns:
            tuple: A tuple containing the next action and the human feedback.
            The human feedback is always GOOD in this implementation.
        """
        return next_action, HumanFeedback.GOOD

    def save_current_step(self, controller_step: ControllerStep, feedback: HumanFeedback = HumanFeedback.GOOD):
        """
        This method is responsible for performing a step in the learning algorithm. And meant to be called
        by the controller after it has performed a step in the environment.

        It first creates a TrajectoryData object from the controller step and the feedback.
        Then it adds the TrajectoryData object to the replay buffer.
        If the episode is finished, it saves the current trajectory in the replay buffer.

        Args:
            controller_step (ControllerStep): The controller step that was performed.
            feedback (HumanFeedback): The feedback for the controller step. Defaults to GOOD.
        """
        step: TrajectoryData = TrajectoryData(
            scene_observation=controller_step.scene_observation,
            action=controller_step.action,
            feedback=torch.Tensor([feedback.value]),
        )

        self._replay_buffer.add(step)
        if controller_step.episode_finished:
            self._replay_buffer.save_current_traj()
            self.save_dataset()

    def reset(self):
        """
        This method is responsible for resetting the current trajectory in the replay buffer.
        Meant to be called by the keyboard observer callback when a trajectory didn't go as planned.
        """
        self._replay_buffer.reset_current_traj()

    @abstractmethod
    def episode_finished(self):
        """
        This method is called when an episode is finished.
        Meant to be called by the controller, so the algorithm can do any necessary cleanup.
        It is an abstract method and needs to be implemented in the subclass.
        """
        pass

    @abstractmethod
    def _training_episode_finished(self):
        """
        This method is a protected method called when a training episode is finished.
        Meant to be called by the train method, so the algorithm can do any necessary cleanup.
        It is an abstract method and needs to be implemented in the subclass.
        """
        pass

    @abstractmethod
    def _action_from_policy(self, scene_observation: SceneObservation):
        """
        This method is responsible for determining the action based on the policy and the current scene observation.

        Not calling directly the policy, because we might need to do some pre-processing before sampling an action.

        Args:
            scene_observation (SceneObservation): The current scene observation.

        It is an abstract method and needs to be implemented in the subclass.
        """
        pass

    def _train_step(self):
        """
        This method is responsible for performing a single training step.

        It first checks if the dataloader is not None. If it is None, it returns immediately.
        Then it gets the next batch from the dataloader and stacks it in a way that allows iteration over time.
        It sets the gradients of all optimized tensors to zero.
        It computes the losses for the batch and calculates the mean loss.
        It performs a backward pass to compute the gradients and updates the parameters using the optimizer.
        It calls the _training_episode_finished method to do any necessary cleanup.
        Finally, it logs the training metrics.
        """
        if self._dataloader is None:
            return
        batch = next(iter(self._dataloader))
        # Batch is a list where each element is a trajectory of size (trajectory_length, feature_size)
        # We need to stack it, but in a way where we can iterate over time, instead of per trajectory.
        # Therefore, we stack it, but in the dimension 1, which results in a tensor of shape
        # (trajectory_length, batch_size, feature_size)
        batch = torch.stack(batch, dim=1)
        self._optimizer.zero_grad()
        losses = self._compute_losses_for_batch(batch)
        total_loss = torch.cat(losses).mean()
        total_loss.backward()
        self._optimizer.step()
        self._training_episode_finished()
        training_metrics = {"loss": total_loss}
        wandb.log(training_metrics)
        if not self._metrics_logger.empty():
            wandb.log(self._metrics_logger.pop())

    def _compute_losses_for_batch(self, trajectories: torch.Tensor):
        """
        This method is responsible for computing the losses for a batch of trajectories.

        It iterates over each time point in the trajectories.
        It moves the time point to the device.
        It creates a tensor filled with the variance value.
        It gets the action from the policy based on the scene observation at the time point.
        It computes the loss between the output and the action at the time point.
        It appends the loss multiplied by the feedback at the time point to the list of losses.
        Finally, it returns the list of losses.

        Args:
            trajectories (torch.Tensor): The batch of trajectories.

        Returns:
            list: The list of losses for the batch of trajectories.
        """
        losses = []
        for time_point in trajectories:
            time_point = time_point.to(device)
            variance = torch.full(time_point.action.size(), 0.1, dtype=torch.float32, device=device)
            out = self._action_from_policy(time_point.scene_observation)
            loss = self._loss_function(out, time_point.action)
            losses.append(loss * time_point.feedback)
        return losses
