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
from utils.timer import Timer
from utils.logging import log_constructor


@dataclass
class CeilingAlgorithmConfig(LearnAlgorithmConfig):
    _ALGO_TYPE: str = field(init=False, default="CeilingAlgorithm")


class CeilingAlgorithm(LearnAlgorithm):
    @log_constructor
    def __init__(
        self,
        config: CeilingAlgorithmConfig,
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

        # Thread to run the training in parallel with the steps as in original CEILing algorithm
        self.__train_thread_running = threading.Event()
        self.__train_thread: Optional[threading.Thread] = None
        # Used to prevent that someone calls the train_step method while it is already running
        # self.__train_step_lock: threading.Lock = threading.Lock()
        super().__init__(config, policy)

    @override
    def train(self, mode: bool = True):
        if mode:
            self.__train_thread_running.set()
            self.__train_thread = threading.Thread(target=self.__train_step)
            self.__train_thread.start()
        else:
            self.__train_thread_running.clear()
            self.__train_thread.join()
            self.__train_thread = None

    @override
    def step(self, controller_step: ControllerStep):
        # TODO get teacher feedback here!!!

        feedback = torch.Tensor([HumanFeedback.GOOD])

        step: TrajectoryData = TrajectoryData(
            scene_observation=controller_step.scene_observation,
            action=controller_step.action,
            feedback=feedback,
        )

        self.__replay_buffer.add(step)
        if controller_step.episode_finished:
            self.__replay_buffer.save_current_traj()

    # Maybe the methods bellow could be moved to the base class as protected methods. Need to check.
    def __train_step(self):
        while self.__train_thread_running.is_set():
            batch = next(iter(self.__dataloader))
            self.__optimizer.zero_grad()
            losses = self.__compute_losses_for_batch(batch)
            total_loss = torch.cat(losses).mean()
            total_loss.backward()
            self.__optimizer.step()
            self._policy.episode_finished()
            training_metrics = {"loss": total_loss}
            wandb.log(training_metrics)

    def __compute_losses_for_batch(self, batch: list):
        losses = []
        lstm_state = None
        for trajectory in batch:
            trajectory = trajectory.to(device)
            variance = torch.full(trajectory.action.size(), 0.1, dtype=torch.float32, device=device)
            out = self._policy([trajectory.scene_observation, lstm_state])
            loss = self.__loss_function(out.squeeze(), trajectory.action, variance)
            losses.append(loss * trajectory.feedback)
        return losses
