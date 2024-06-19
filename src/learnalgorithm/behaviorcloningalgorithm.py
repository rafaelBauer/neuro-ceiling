import threading
from dataclasses import dataclass, field
from typing import Optional

import torch
import wandb
from overrides import override
from torch.utils.data import RandomSampler, DataLoader

from controller.controllerstep import ControllerStep
from learnalgorithm.learnalgorithm import LearnAlgorithmConfig, LearnAlgorithm
from policy import PolicyBase
from utils.dataset import TrajectoriesDataset, TrajectoryData
from utils.device import device
from utils.human_feedback import HumanFeedback
from utils.logging import log_constructor


@dataclass
class BehaviorCloningAlgorithmConfig(LearnAlgorithmConfig):
    _ALGO_TYPE: str = field(init=False, default="BehaviorCloningAlgorithm")
    number_of_epochs: int = field(init=True)


class BehaviorCloningAlgorithm(LearnAlgorithm):
    @log_constructor
    def __init__(
        self,
        config: BehaviorCloningAlgorithmConfig,
        policy: PolicyBase,
    ):

        # Replay buffer and Data Loader
        self.__replay_buffer: TrajectoriesDataset = torch.load(config.dataset_path)
        self.__sampler = RandomSampler(self.__replay_buffer)
        self.__dataloader: DataLoader = DataLoader(
            self.__replay_buffer, sampler=self.__sampler, batch_size=config.batch_size, collate_fn=lambda x: x
        )

        # Which loss function to use for the algorithm
        self.__loss_function = torch.nn.GaussianNLLLoss()

        # Optimizer
        self.__optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        super().__init__(config, policy)

    @override
    def train(self, mode: bool = True):
        if mode:
            for _ in range(self._CONFIG.number_of_epochs):
                batch = next(iter(self.__dataloader))
                # Batch is a list where each element is a trajectory of size (trajectory_length, feature_size)
                # We need to convert it to a tensor of shape (trajectory_length, batch_size, feature_size)
                # Since the training is done per batch, and we need to iterate over time
                batch = torch.stack(batch, dim=1)
                self.__optimizer.zero_grad()
                losses = self.__compute_losses_for_batch(batch)
                total_loss = torch.cat(losses).mean()
                total_loss.backward()
                self.__optimizer.step()
                self._policy.episode_finished()
                training_metrics = {"loss": total_loss}
                wandb.log(training_metrics)

    @override
    def step(self, controller_step: ControllerStep):
        feedback = torch.Tensor([HumanFeedback.GOOD])

        step: TrajectoryData = TrajectoryData(
            scene_observation=controller_step.scene_observation,
            action=controller_step.action,
            feedback=feedback,
        )

        self.__replay_buffer.add(step)
        if controller_step.episode_finished:
            self.__replay_buffer.save_current_traj()

    def __compute_losses_for_batch(self, trajectories: torch.Tensor):
        losses = []
        for time_point in trajectories:
            time_point = time_point.to(device)
            variance = torch.full(time_point.action.size(), 0.1, dtype=torch.float32, device=device)
            out = self._policy(time_point.scene_observation)
            loss = self.__loss_function(out, time_point.action, variance)
            losses.append(loss * time_point.feedback)
        return losses

    def reset(self):
        self.__replay_buffer.reset_current_traj()