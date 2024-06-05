from abc import abstractmethod


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
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def get_action_sequence(self):
        """
        An abstract method that must be implemented in any subclass. It should return a sequence of actions to perform the goal.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
