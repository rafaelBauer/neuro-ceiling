import logging

import numpy as np
import pylsl

from neuroceiling import KeyboardObserver


class GamepadToLSL:
    def __init__(self):
        self.__gamepad: KeyboardObserver = KeyboardObserver()
        self.__stream_outlet: pylsl.StreamOutlet = pylsl.StreamOutlet(pylsl.StreamInfo("Gamepad", "markers",
                                                                                       8, pylsl.IRREGULAR_RATE,
                                                                                       pylsl.cf_float32, "Gamepad", ))

    def start(self) -> None:
        self.__gamepad.subscribe_callback_to_gripper(self.__gripper_callback)
        self.__gamepad.subscribe_callback_to_label(self.__label_callback)
        self.__gamepad.subscribe_callback_to_direction(self.__direction_callback)
        self.__gamepad.subscribe_callback_to_reset(self.__reset_callback)
        self.__gamepad.start()

    def stop(self) -> None:
        self.__gamepad.unsubscribe_callback_to_gripper(self.__gripper_callback)
        self.__gamepad.unsubscribe_callback_to_label(self.__label_callback)
        self.__gamepad.unsubscribe_callback_to_direction(self.__direction_callback)
        self.__gamepad.stop()

    def __gripper_callback(self, gripper: float) -> None:
        print("Gripper: " + gripper.__str__())
        data = [self.__gamepad.label, gripper]
        data.extend(self.__gamepad.direction)
        self.__stream_outlet.push_sample(data)

    def __label_callback(self, label: int) -> None:
        print("Label: " + label.__str__())
        data = [label, self.__gamepad.gripper]
        data.extend(self.__gamepad.direction)
        self.__stream_outlet.push_sample(data)

    def __direction_callback(self, direction: [float]) -> None:
        print("Direction: " + direction.__str__())
        data = [self.__gamepad.label, self.__gamepad.gripper]
        data.extend(direction)
        self.__stream_outlet.push_sample(data)

    def __reset_callback(self) -> None:
        print("Reset requested")


def main() -> None:
    logging.basicConfig()
    logging.root.setLevel(logging.NOTSET)

    gamepad_to_lsl: GamepadToLSL = GamepadToLSL()
    gamepad_to_lsl.start()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Stopping stream...")
        gamepad_to_lsl.stop()


if __name__ == '__main__':
    main()
