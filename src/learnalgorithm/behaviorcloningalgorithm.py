from dataclasses import dataclass, field

import torch
from overrides import override
from torch.utils.data import RandomSampler, DataLoader
from tqdm import tqdm

from envs.robotactions import RobotAction
from goal.goal import Goal
from learnalgorithm.learnalgorithm import LearnAlgorithmConfig, LearnAlgorithm
from policy import PolicyBase
from utils.human_feedback import HumanFeedback
from utils.logging import log_constructor
from utils.sceneobservation import SceneObservation


@dataclass
class BehaviorCloningAlgorithmConfig(LearnAlgorithmConfig):
    _ALGO_TYPE: str = field(init=False, default="BehaviorCloningAlgorithm")
    number_of_epochs: int = field(init=True, default=0)


class BehaviorCloningAlgorithm(LearnAlgorithm):
    @log_constructor
    def __init__(self, config: BehaviorCloningAlgorithmConfig, policy: PolicyBase, feedback_device):

        # Which loss function to use for the algorithm
        loss_function = torch.nn.CrossEntropyLoss()

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
            with tqdm(total=self._CONFIG.number_of_epochs, desc="Training ") as progress_bar:
                for _ in range(self._CONFIG.number_of_epochs):
                    self._train_step()
                    progress_bar.update(1)

    @override
    def get_human_feedback(
        self, next_action: Goal | RobotAction, scene_observation: SceneObservation
    ) -> (Goal | RobotAction, HumanFeedback):
        return HumanFeedback.GOOD

    @override()
    def _training_episode_finished(self):
        self._policy.episode_finished()

    @override
    def _action_from_policy(self, scene_observation: SceneObservation) -> torch.Tensor:
        return self._policy(scene_observation)

    @override
    def episode_finished(self):
        pass
