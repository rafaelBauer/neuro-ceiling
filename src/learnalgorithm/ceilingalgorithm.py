import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy
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
from utils.metricslogger import MetricsLogger, EpisodeMetrics
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
        self._feedback_device.subscribe_callback_to_direction(self.__key_pressed_callback)

        # Thread to run the training in parallel with the steps as in original CEILing algorithm
        self.__train_thread_running = threading.Event()
        self.__train_thread: Optional[threading.Thread] = None

        self.__lstm_state = None

        self.__last_feedback: numpy.array = numpy.zeros(4)

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
        if numpy.any(self.__last_feedback):
            # TODO: This is a hardcoded label algorithm for now
            label = torch.zeros(4)

            # Conversion from keyboard_observer to label. Needs to be changed so EEG output would also work
            if self.__last_feedback[5] > 0.5:  # "o" key
                label[0] = True
            elif self.__last_feedback[4] < -0.5:  # "i" key
                label[1] = True
            elif self.__last_feedback[5] < -0.5:  # "u" key
                label[2] = True
            else:
                label[3] = True

            self.__last_feedback = numpy.zeros(self.__last_feedback.size)

            action = PickPlaceObject.from_tensor(label, scene_observation)
            if action != next_action:
                feedback = HumanFeedback.CORRECTED
            else:
                feedback = HumanFeedback.GOOD
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

    def __key_pressed_callback(self, action: numpy.array):
        """
        This function is a callback that is triggered when a key is pressed.

        It checks if any action has been performed. If not, it returns and does nothing.
        If an action has been performed, it updates the last action and sets the new command flag to True.

        Args:
            action (numpy.array): An array representing the action performed.
        """
        if not numpy.any(action):
            return
        self.__last_feedback = action
