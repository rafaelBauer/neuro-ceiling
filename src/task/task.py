from abc import abstractmethod


class Task:
    def __init__(self):
        pass

    @abstractmethod
    def get_action_sequence(self):
        raise NotImplementedError("Must be implemented in subclass")
