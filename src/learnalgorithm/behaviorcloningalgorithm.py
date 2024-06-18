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

    def __compute_losses_for_batch(self, batch: list):
        losses = []
        for trajectory in batch:
            trajectory = trajectory.to(device)
            variance = torch.full(trajectory.action.size(), 0.1, dtype=torch.float32, device=device)
            out = self._policy(trajectory.scene_observation)
            loss = self.__loss_function(out.squeeze(), trajectory.action, variance)
            losses.append(loss * trajectory.feedback)
        return losses
