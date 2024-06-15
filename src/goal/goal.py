from abc import abstractmethod

import torch
from tensordict import tensorclass


class Goal:
    """
    This is an abstract base class that represents a generic goal.

    It can be processed by a controller. The Controller will receive a Goad and will have to perform a sequence of
    actions to complete it.

    Methods:
        get_action_sequence: An abstract method that must be implemented in any subclass. It should return a sequence
        of actions to perform the goal.
    """

    def __init__(self):
        """
        The constructor for goal class. It doesn't take any parameters.
        """
        empty: torch.Tensor = torch.Tensor(0)

    @abstractmethod
    def __eq__(self, other) -> bool:
        """
        Method that must be implemented in any subclass. It should return a boolean indicating if the goal
        is equal to another goal.

        In this case, since it's a "Null Object", it checks if it's the same instance by verifying the address.
        """
        return id(self) == id(other)

    @abstractmethod
    def get_action_sequence(self):
        """
        An abstract method that must be implemented in any subclass. It should return a sequence of actions to perform the goal.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        return []

    @abstractmethod
    def to_tensor(self):
        """
        Method to convert the goal to a tensor.

        Returns:
            torch.Tensor: A tensor representing the goal.
        """
        return torch.tensor([])
