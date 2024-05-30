from abc import abstractmethod


class Task:
    """
    This is an abstract base class that represents a generic task.

    It can be processed by an Agent. The Agent will receive a Task and will have to perform a sequence of actions to complete it.

    Methods:
        get_action_sequence: An abstract method that must be implemented in any subclass. It should return a sequence of actions to perform the task.
    """

    def __init__(self):
        """
        The constructor for Task class. It doesn't take any parameters.
        """
        pass

    @abstractmethod
    def get_action_sequence(self):
        """
        An abstract method that must be implemented in any subclass. It should return a sequence of actions to perform the task.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Must be implemented in subclass")
