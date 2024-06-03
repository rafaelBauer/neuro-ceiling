from abc import abstractmethod
from dataclasses import dataclass, field

from torch import nn, Tensor

from goal.goal import Goal
from utils.logging import log_constructor


@dataclass(kw_only=True)
class PolicyBaseConfig:
    _POLICY_TYPE: str = field(init=True)

    @property
    def policy_type(self) -> str:
        return self._POLICY_TYPE


class PolicyBase(nn.Module):
    @log_constructor
    def __init__(self, config: PolicyBaseConfig, **kwargs):
        # Deleting unnecessary kwargs from children classes
        if "environment" in kwargs:
            del kwargs["environment"]
        if "keyboard_observer" in kwargs:
            del kwargs["keyboard_observer"]

        self._CONFIG = config

        super().__init__(**kwargs)

    @abstractmethod
    def update(self):
        """
        Method usually called by the learning algorithm to update the policy.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("The update method must be implemented in a subclass.")

    @abstractmethod
    def forward(self, states: Tensor) -> Tensor:
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
    def task_to_be_executed(self, goal: Goal):
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