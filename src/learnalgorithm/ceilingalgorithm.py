import threading
from dataclasses import dataclass, field
from typing import Optional

import torch
from overrides import override
from torch.utils.data import RandomSampler, DataLoader

from controller.controllerstep import ControllerStep
from envs.robotactions import RobotAction
from goal import PickPlaceObject
from goal.goal import Goal
from learnalgorithm.learnalgorithm import LearnAlgorithmConfig, LearnAlgorithm
from policy import PolicyBase
from utils.gripperstate import GripperState
from utils.human_feedback import HumanFeedback
from utils.keyboard_observer import KeyboardObserver
from utils.labeltoobjectpose import LabelToPoseTranslator
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
        feedback_device,
    ):

        # Which loss function to use for the algorithm
        self.__label_to_goal_translator = LabelToPoseTranslator()
        loss_function = torch.nn.CrossEntropyLoss()

        # Optimizer
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        super().__init__(config, policy, RandomSampler, DataLoader, loss_function, optimizer)

        # TODO: for now have this as a keyboard observer,
        #       but should be a feedback device which could also come from EEG
        self._feedback_device: KeyboardObserver = feedback_device

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
    def get_human_feedback(
        self, next_action: Goal | RobotAction, scene_observation: SceneObservation
    ) -> (Goal | RobotAction, HumanFeedback):
        action = next_action
        if self._feedback_device.is_direction_commanded:
            feedback_device_output = self._feedback_device.direction

            # TODO: This is a hardcoded label for now
            label = torch.zeros(4)

            # Conversion from keyboard_observer to label. Needs to be changed so EEG output would also work
            if feedback_device_output[5] < -0.5:  # "u" key
                label[0] = True
            elif feedback_device_output[4] < -0.5:  # "i" key
                label[1] = True
            elif feedback_device_output[5] > 0.5:  # "o" key
                label[2] = True
            else:
                label[3] = True

            action = PickPlaceObject.from_label_tensor(label, scene_observation)
            feedback = HumanFeedback.CORRECTED
        else:
            feedback = self._feedback_device.label
        return action, feedback

    @override
    def _training_episode_finished(self):
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

    @override
    def episode_finished(self):
        self._feedback_device.reset()
