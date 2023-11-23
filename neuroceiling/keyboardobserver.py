"""
Class copied from CEILing project.
https://github.com/robot-learning-freiburg/CEILing/blob/main/src/utils.py
"""
import numpy as np
from pynput import keyboard
from functools import partial
import time


class KeyboardObserver:
    def __init__(self):
        self.reset_button = False
        self.reset()
        self.hotkeys = keyboard.GlobalHotKeys(
            {
                "g": partial(self.set_label, 1),  # good
                "b": partial(self.set_label, 0),  # bad
                "c": partial(self.set_gripper, -0.9),  # close
                "v": partial(self.set_gripper, 0.9),  # open
                "f": partial(self.set_gripper, None),  # gripper free
                "x": self.reset_episode,
            }
        )
        self.hotkeys.start()

        # Need to sleep because apparently there is a race condition problem with one dependent
        # library from pynput (pyobjc).
        # The solution is described in this comment:
        # https://github.com/moses-palmer/pynput/issues/55#issuecomment-1314410235
        time.sleep(1)

        self.direction = np.array([0, 0, 0, 0, 0, 0])
        self.listener = keyboard.Listener(
            on_press=self.set_direction, on_release=self.reset_direction
        )
        self.key_mapping = {
            "a": (1, 1),  # left
            "d": (1, -1),  # right
            "s": (0, 1),  # backward
            "w": (0, -1),  # forward
            "q": (2, 1),  # down
            "e": (2, -1),  # up
            "j": (3, -1),  # look left
            "l": (3, 1),  # look right
            "i": (4, -1),  # look up
            "k": (4, 1),  # look down
            "u": (5, -1),  # rotate left
            "o": (5, 1),  # rotate right
        }
        self.listener.start()
        return

    def set_label(self, value):
        self.label = value
        print("label set to: ", value)
        return

    def get_label(self):
        return self.label

    def set_gripper(self, value):
        self.gripper_open = value
        print("gripper set to: ", value)
        return

    def get_gripper(self):
        return self.gripper_open

    def set_direction(self, key):
        try:
            idx, value = self.key_mapping[key.char]
            self.direction[idx] = value
        except (KeyError, AttributeError):
            pass
        return

    def reset_direction(self, key):
        try:
            idx, _ = self.key_mapping[key.char]
            self.direction[idx] = 0
        except (KeyError, AttributeError):
            pass
        return

    def has_joints_cor(self):
        return self.direction.any()

    def has_gripper_update(self):
        return self.get_gripper() is not None

    def get_ee_action(self):
        return self.direction * 0.9

    def reset_episode(self):
        self.reset_button = True
        return

    def reset(self):
        self.set_label(1)
        self.set_gripper(None)
        self.reset_button = False
        return
