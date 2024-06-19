from dataclasses import dataclass, field

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
from utils.sceneobservation import SceneObservation


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

        # Which loss function to use for the algorithm
        loss_function = torch.nn.GaussianNLLLoss()

        # Optimizer
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        super().__init__(config, policy, RandomSampler, DataLoader, loss_function, optimizer)

    @override
    def train(self, mode: bool = True):
        if mode:
            for _ in range(self._CONFIG.number_of_epochs):
                self._train_step()

    @override
    def _get_human_feedback(self, controller_step: ControllerStep):
        return torch.Tensor([HumanFeedback.GOOD])

    def _episode_finished(self):
        self._policy.episode_finished()

    def _action_from_policy(self, scene_observation: SceneObservation) -> torch.Tensor:
        return self._policy(scene_observation)

    def reset(self):
        self.__replay_buffer.reset_current_traj()
