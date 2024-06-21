from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Final, TypeVar, Type, Optional

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
from utils.sceneobservation import SceneObservation


@dataclass
class LearnAlgorithmConfig:
    _ALGO_TYPE: str = field(init=False)
    batch_size: int = field(init=True)
    learning_rate: float = field(init=True)
    weight_decay: float = field(init=True)
    steps_per_episode: int = field(init=True)
    load_dataset: str = field(init=True, default="")
    save_dataset: str = field(init=True, default="")

    @property
    def algo_type(self) -> str:
        return self._ALGO_TYPE


@dataclass
class NoLearnAlgorithmConfig(LearnAlgorithmConfig):
    _ALGO_TYPE: str = field(init=False, default="NoLearnAlgorithm")
    batch_size: int = field(init=False, default=0)
    learning_rate: float = field(init=False, default=0)
    weight_decay: float = field(init=False, default=0)
    steps_per_episode: int = field(init=False, default=0)


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

        policy.to(device)
        wandb.watch(policy, log_freq=100)

    def load_dataset_from_file(self):
        if self._CONFIG.load_dataset:
            self._replay_buffer: TrajectoriesDataset = torch.load(self._CONFIG.load_dataset)
            self._sampler = self.__sampler_type(self._replay_buffer)
            self._dataloader = self.__dataloader_type(
                dataset=self._replay_buffer,
                sampler=self._sampler,
                batch_size=self._CONFIG.batch_size,
                collate_fn=lambda x: x,
            )

    def save_dataset_to_file(self):
        if self._CONFIG.save_dataset:
            torch.save(self._replay_buffer, self._CONFIG.save_dataset)
            logger.info("Successfully saved dataset {}", self._CONFIG.save_dataset)

    @abstractmethod
    def train(self, mode: bool = True):
        pass

    def get_human_feedback(
        self, next_action: Goal | RobotAction, scene_observation: SceneObservation
    ) -> (Goal | RobotAction, HumanFeedback):
        return next_action, HumanFeedback.GOOD

    def step(self, controller_step: ControllerStep, feedback: HumanFeedback = HumanFeedback.GOOD):
        step: TrajectoryData = TrajectoryData(
            scene_observation=controller_step.scene_observation,
            action=controller_step.action,
            feedback=torch.Tensor([feedback.value]),
        )

        self._replay_buffer.add(step)
        if controller_step.episode_finished:
            self._replay_buffer.save_current_traj()

    @abstractmethod
    def reset(self):
        self._replay_buffer.reset_current_traj()

    @abstractmethod
    def episode_finished(self):
        pass

    @abstractmethod
    def _training_episode_finished(self):
        pass

    @abstractmethod
    def _action_from_policy(self, scene_observation: SceneObservation):
        pass

    def _train_step(self):
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

    def _compute_losses_for_batch(self, trajectories: torch.Tensor):
        losses = []
        for time_point in trajectories:
            time_point = time_point.to(device)
            variance = torch.full(time_point.action.size(), 0.1, dtype=torch.float32, device=device)
            out = self._action_from_policy(time_point.scene_observation)
            loss = self._loss_function(out, time_point.action)
            losses.append(loss * time_point.feedback)
        return losses
