import os
from abc import abstractmethod
from dataclasses import dataclass, field

import torch
import wandb
from torch import nn, Tensor

from goal.goal import Goal
from utils.logging import log_constructor, logger
from utils.sceneobservation import SceneObservation


@dataclass(kw_only=True)
class PolicyBaseConfig:
    """
    The PolicyBaseConfig class represents the base configuration for a policy.

    Attributes:
        _POLICY_TYPE (str): The type of the policy.
    """

    _POLICY_TYPE: str = field(init=True)
    from_file: str = field(init=True, default="")
    save_to_file: str = field(init=True, default="")

    @property
    def policy_type(self) -> str:
        """
        The policy_type property of the policy base configuration.

        Returns:
            str: The type of the policy.
        """
        return self._POLICY_TYPE


class PolicyBase(nn.Module):
    """
    The PolicyBase class is the base for every policy.

    Attributes:
        _CONFIG (PolicyBaseConfig): The configuration for the policy base.
    """

    @log_constructor
    def __init__(self, config: PolicyBaseConfig, **kwargs):
        # Deleting unnecessary kwargs from children classes
        if "environment" in kwargs:
            del kwargs["environment"]
        if "keyboard_observer" in kwargs:
            del kwargs["keyboard_observer"]

        self._CONFIG = config

        super().__init__(**kwargs)

    def load_from_file(self):
        if self._CONFIG.from_file:
            try:
                file_name_and_extension = os.path.basename(self._CONFIG.from_file)
                task_name = os.path.dirname(self._CONFIG.from_file).split("/")[-1]
                artifact = wandb.run.use_artifact(f"{task_name}/{os.path.splitext(file_name_and_extension)[0]}:latest")
                model_dir = artifact.download()
                model_file = os.path.join(model_dir, file_name_and_extension)
            except wandb.CommError as exception:
                logger.info("Could not download artifact from wandb: {}", exception)
                logger.info(f"Using policy from  local filesystem: {self._CONFIG.from_file}")
                model_file = self._CONFIG.from_file

            logger.info("Loading policy from file: {}", model_file)
            self.load_state_dict(torch.load(model_file))

    def save_to_file(self):
        if self._CONFIG.save_to_file:
            logger.info(f"Saving policy to file: {self._CONFIG.save_to_file}")
            torch.save(self.state_dict(), self._CONFIG.save_to_file)

    def publish_model(self):
        if self._CONFIG.save_to_file:
            self.save_to_file()
            logger.info(f"Publishing policy to wandb: {self._CONFIG.save_to_file}")
            file_name_and_extension = os.path.basename(self._CONFIG.save_to_file)
            artifact = wandb.Artifact(f"{os.path.splitext(file_name_and_extension)[0]}", type='model')
            artifact.add_file(self._CONFIG.save_to_file)
            wandb.run.log_artifact(artifact)

    @abstractmethod
    def forward(self, states) -> Tensor:
        """
        Method that defines the forward pass of the policy.
        In the forward function we accept a Tensor of input data, and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary (differentiable) operations on Tensors.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        """
        raise NotImplementedError("The forward method must be implemented in a subclass.")

    @abstractmethod
    def goal_to_be_achieved(self, goal: Goal):
        """
        Method that will cause the policy to know which goal the has to be executed by the controller. It might
        use this information to adjust itself, or simply ignore this information.

        This method could potentially cause a race condition if it is called from a different thread than the
        forward method. So it must be implemented in a way that it is thread-safe.

        Parameters:
            goal (Goal): The goal to be planned.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("The task_to_be_executed method must be implemented in a subclass.")

    @abstractmethod
    def episode_finished(self):
        """
        Method that will cause the policy to know that the episode has finished. It might
        use this information to adjust itself, or simply ignore this information.
        """
        raise NotImplementedError("The episode_finished method must be implemented in a subclass.")
