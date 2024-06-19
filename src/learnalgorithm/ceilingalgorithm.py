import threading
from dataclasses import dataclass, field
from typing import Optional

import torch
from overrides import override
from torch.utils.data import RandomSampler, DataLoader

from controller.controllerstep import ControllerStep
from learnalgorithm.learnalgorithm import LearnAlgorithmConfig, LearnAlgorithm
from policy import PolicyBase
from utils.human_feedback import HumanFeedback
from utils.logging import log_constructor
from utils.sceneobservation import SceneObservation


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

        # Which loss function to use for the algorithm
        loss_function = torch.nn.GaussianNLLLoss()

        # Optimizer
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        super().__init__(config, policy, RandomSampler, DataLoader, loss_function, optimizer)

        # Thread to run the training in parallel with the steps as in original CEILing algorithm
        self.__train_thread_running = threading.Event()
        self.__train_thread: Optional[threading.Thread] = None

        self.__lstm_state = None

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
    def _get_human_feedback(self, controller_step: ControllerStep):
        return torch.Tensor([HumanFeedback.GOOD])

    @override
    def _episode_finished(self):
        self.__lstm_state = None

    @override
    def _action_from_policy(self, scene_observation: SceneObservation) -> torch.Tensor:
        policy_input = [scene_observation, self.__lstm_state]
        out = self._policy(policy_input)
        self.__lstm_state = policy_input[1]
        return out

    def __train_step(self):
        while self.__train_thread_running.is_set():
            self._train_step()

    def reset(self):
        self._replay_buffer.reset_current_traj()
