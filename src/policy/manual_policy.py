from typing import Final

from policy.policy import PolicyBase
from utils.keyboard_observer import KeyboardObserver


class ManualPolicy(PolicyBase):
    def __init__(self, keyboard_observer: KeyboardObserver, **kwargs):
        super(ManualPolicy, self).__init__(**kwargs)
        self.__keyboard_observer: Final[KeyboardObserver] = keyboard_observer
