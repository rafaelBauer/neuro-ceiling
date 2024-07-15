import threading
from abc import abstractmethod

import torch


class Goal:
    """
    This is an abstract base class that represents a generic goal.

    It can be processed by a controller. The Controller will receive a Goad and will have to perform a sequence of
    actions to complete it.

    Methods:
        get_action_sequence: An abstract method that must be implemented in any subclass. It should return a sequence
        of actions to perform the goal.
    """

    def __init__(self, length: int = 0, input_tensor: torch.Tensor = None):
        """
        The constructor for goal class. It doesn't take any parameters.

        Args:
            length (int): The length of the tensor representing the goal. Defaults to 0.
        """
        if input_tensor is not None:
            self.__tensor: torch.Tensor = input_tensor
        else:
            self.__tensor: torch.Tensor = torch.zeros(length)
        self.__completed: bool = False
        self.__completed_lock: threading.Lock = threading.Lock()

    @abstractmethod
    def __eq__(self, other) -> bool:
        """
        Method that must be implemented in any subclass. It should return a boolean indicating if the goal
        is equal to another goal.

        In this case, since it's a "Null Object", it checks if it's the same instance by verifying the address.

        Args:
            other (Goal): Another Goal instance to compare with.

        Returns:
            bool: True if the two Goal instances are the same, False otherwise.
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
        return self.__tensor

    @abstractmethod
    def replaceable(self, other: "Goal") -> bool:
        """
        Method to check if the goal is replaceable by another goal.
        """
        return True

    @property
    def is_completed(self):
        """
        Method to check if the goal is completed.

        Returns:
            bool: True if the goal is completed, False otherwise.
        """
        with self.__completed_lock:
            return self.__completed

    def complete(self):
        """
        Method to tell that the goal has been completed.
        """
        with self.__completed_lock:
            self.__completed = True
