import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy
import torch
import wandb
from overrides import override
from torch.utils.data import RandomSampler, DataLoader

from envs.robotactions import RobotAction
from goal import PickPlaceObject
from goal.goal import Goal
from learnalgorithm.feedbackdevice.feedbackdevice import FeedbackDevice, FeedbackDeviceConfig
from learnalgorithm.learnalgorithm import LearnAlgorithmConfig, LearnAlgorithm
from policy import PolicyBase
from utils.human_feedback import HumanFeedback
from utils.labeltoobjectpose import LabelToPoseTranslator
from utils.logging import log_constructor, logger
from utils.sceneobservation import SceneObservation


@dataclass
class CeilingAlgorithmConfig(LearnAlgorithmConfig):
    _ALGO_TYPE: str = field(init=False, default="CeilingAlgorithm")
    feedback_device_config: FeedbackDeviceConfig = field(init=True, default_factory=FeedbackDeviceConfig(action_dim=0))
    load_dataset: str = field(init=True, default="")
    save_dataset: str = field(init=True, default="")

    @property
    @override
    def name(self) -> str:
        return "ceiling"


class CeilingAlgorithm(LearnAlgorithm):
    @log_constructor
    def __init__(
        self,
        config: CeilingAlgorithmConfig,
        policy: PolicyBase,
        feedback_device: FeedbackDevice,
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

        self._feedback_device: FeedbackDevice = feedback_device
        self._feedback_device.subscribe_callback_to_evaluative_feedback(self.__feedback_device_label_callback)

        # Thread to run the training in parallel with the steps as in original CEILing algorithm
        self.__train_thread_running = threading.Event()
        self.__train_thread: Optional[threading.Thread] = None

        self.__lstm_state = None
        wandb.run.tags = ["CEILing"]

    @override
    def train(self, mode: bool = True):
        if mode:
            self.__train_thread_running.set()
            self.__train_thread = threading.Thread(target=self.__train_step)
            self.__train_thread.start()
        else:
            if self.__train_thread_running.is_set():
                self.__train_thread_running.clear()
                self.__train_thread.join()
                self.__train_thread = None

    @override
    def get_human_feedback(
        self, next_action: Goal | RobotAction, scene_observation: SceneObservation
    ) -> (Goal | RobotAction, HumanFeedback):
        action = next_action
        action_tensor = next_action.to_tensor()
        corrected_action_tensor, evaluative_feedback = self._feedback_device.sample_feedback(
            action_tensor, scene_observation
        )
        if evaluative_feedback == HumanFeedback.CORRECTED:
            action = PickPlaceObject.from_tensor(corrected_action_tensor, scene_observation)
            logger.debug(f"Corrected action:\n" f"     original: {next_action}\n" f"     corrected: {action}")
        return action, evaluative_feedback

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

    def __feedback_device_label_callback(self, label: HumanFeedback):
        """
        This function is a callback that is triggered when a label is received from the feedback device.

        It updates the feedback label.

        Args:
            label (HumanFeedback): The label received.
        """
        if label == HumanFeedback.BAD:
            self._replay_buffer.modify_feedback_from_current_step(label)
            self._feedback_update_callback(label)
        return
