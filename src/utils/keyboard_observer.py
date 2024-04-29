"""
Class based from CEILing project. Some extra features were added and functions renamed.
https://github.com/robot-learning-freiburg/CEILing/blob/main/src/utils.py
"""
from __future__ import annotations
from enum import Enum
from collections.abc import Callable
from functools import partial
import time
import threading
from typing import Final

import numpy as np
from pynput import keyboard


class HumanFeedback(Enum):
    """
    Enum class for human feedback.

    Attributes:
        CORRECTED: A human had to correct the action. Value is -1.
        BAD: A human considered a bad action. Value is 0.
        GOOD: A human considered a good action Value is 1.
    """

    CORRECTED = -1
    BAD = 0
    GOOD = 1


class KeyboardObserver:
    """
    This class is used to observe keyboard inputs and map them to certain actions.
    """

    # Key mapping consists of (i, j), where:
    #   i = The index in the direction array
    #   j = The value of that index in the direction array
    __KEY_MAPPING: Final[dict[str, tuple[int, int]]] = {
        "s": (0, 1),  # backward
        "w": (0, -1),  # forward
        "a": (1, 1),  # left
        "d": (1, -1),  # right
        "q": (2, 1),  # down
        "e": (2, -1),  # up
        "j": (3, -1),  # look left
        "l": (3, 1),  # look right
        "i": (4, -1),  # look up
        "k": (4, 1),  # look down
        "u": (5, -1),  # rotate left
        "o": (5, 1),  # rotate right
    }

    def __init__(self):
        self.__label: HumanFeedback = HumanFeedback.GOOD
        self.__gripper: float | None = None
        self.__direction: np.array = np.zeros(6)

        self.__label_lock: threading.Lock = threading.Lock()
        self.__gripper_lock: threading.Lock = threading.Lock()
        self.__direction_lock: threading.Lock = threading.Lock()
        self.__hotkeys = keyboard.GlobalHotKeys(
            {
                "g": partial(self.__set_label, HumanFeedback.GOOD),
                "b": partial(self.__set_label, HumanFeedback.BAD),  # bad
                "c": partial(self.__set_gripper, -0.9),  # close
                "v": partial(self.__set_gripper, 0.9),  # open
                "f": partial(self.__set_gripper, None),  # gripper free
                "x": self.reset_episode,  # reset
            }
        )
        self.__listener = keyboard.Listener(
            on_press=self.__set_direction, on_release=self.__reset_direction
        )
        self.__direction_keys_pressed: set = set()  # store which keys are pressed

        self.__label_callbacks: [Callable[[int], None]] = []
        self.__gripper_callbacks: [Callable[[float], None]] = []
        self.__direction_callbacks: [Callable[[np.array], None]] = []
        self.__reset_callbacks: [Callable[[], None]] = []

    def start(self) -> None:
        self.__hotkeys.start()

        # Need to sleep because apparently there is a race condition problem with one dependent
        # library from pynput when running on macOS (pyobjc).
        # The solution is described in this comment:
        # https://github.com/moses-palmer/pynput/issues/55#issuecomment-1314410235
        time.sleep(1)
        self.__listener.start()

    def stop(self) -> None:
        self.__hotkeys.stop()
        self.__listener.stop()

    def __set_label(self, value: int) -> None:
        with self.__label_lock:
            self.__label = value
            self.__call_callbacks(self.__label_callbacks, self.__label)

    def subscribe_callback_to_label(self, callback_func: Callable[[int], None]) -> None:
        with self.__label_lock:
            self.__label_callbacks.append(callback_func)

    def unsubscribe_callback_to_label(
        self, callback_func: Callable[[int], None]
    ) -> None:
        with self.__label_lock:
            self.__label_callbacks.remove(callback_func)

    @property
    def label(self) -> HumanFeedback:
        with self.__label_lock:
            return self.__label

    def __set_gripper(self, value: float | None) -> None:
        with self.__gripper_lock:
            self.__gripper = value
            self.__call_callbacks(self.__gripper_callbacks, self.__gripper)

    def subscribe_callback_to_gripper(
        self, callback_func: Callable[[float], None]
    ) -> None:
        with self.__gripper_lock:
            self.__gripper_callbacks.append(callback_func)

    def unsubscribe_callback_to_gripper(
        self, callback_func: Callable[[float], None]
    ) -> None:
        with self.__gripper_lock:
            self.__gripper_callbacks.remove(callback_func)

    @property
    def gripper(self) -> float | None:
        with self.__gripper_lock:
            return self.__gripper

    @property
    def is_gripper_commanded(self) -> bool:
        """
        Method to verify weather there is any active gripper command
        :return: If there is any gripper direction command, otherwise False
        """
        with self.__gripper_lock:
            return self.__gripper is not None

    def __set_direction(self, key) -> None:
        with self.__direction_lock:
            try:
                idx, value = self.__KEY_MAPPING[key.char]
                if key not in self.__direction_keys_pressed:
                    self.__direction[idx] = value
                    self.__call_callbacks(self.__direction_callbacks, self.__direction)
                    self.__direction_keys_pressed.add(key)
            except (KeyError, AttributeError):
                pass

    def subscribe_callback_to_direction(
        self, callback_func: Callable[[np.array], None]
    ) -> None:
        with self.__direction_lock:
            self.__direction_callbacks.append(callback_func)

    def unsubscribe_callback_to_direction(
        self, callback_func: Callable[[np.array], None]
    ) -> None:
        with self.__direction_lock:
            self.__direction_callbacks.remove(callback_func)

    def __reset_direction(self, key) -> None:
        with self.__direction_lock:
            try:
                idx, _ = self.__KEY_MAPPING[key.char]
                old_direction_array = self.__direction.copy()
                self.__direction[idx] = 0
                if not np.array_equal(old_direction_array, self.__direction):
                    self.__call_callbacks(self.__direction_callbacks, self.__direction)
                self.__direction_keys_pressed.remove(key)
            except (KeyError, AttributeError):
                pass

    @property
    def direction(self) -> np.array:
        with self.__direction_lock:
            return self.__direction

    @property
    def is_direction_commanded(self) -> bool:
        """
        Method to verify weather there is any active direction command
        :return: If there is any active direction command, otherwise False
        """
        with self.__direction_lock:
            return np.count_nonzero(self.__direction) != 0

    def get_ee_action(self) -> np.array:
        with self.__direction_lock:
            return self.direction * 0.9

    def reset_episode(self) -> None:
        for callback in self.__reset_callbacks:
            callback()
        self.__set_label(1)
        self.__set_gripper(0)

    def subscribe_callback_to_reset(self, callback_func: Callable[[], None]) -> None:
        self.__reset_callbacks.append(callback_func)

    @classmethod
    def __call_callbacks(cls, callbacks, value) -> None:
        if len(callbacks) != 0:
            for callback in callbacks:
                callback(value)