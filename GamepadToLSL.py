import logging
import pylsl

from neuroceiling import KeyboardObserver


class GamepadToLSL:
    def __init__(self):
        self.__gamepad: KeyboardObserver = KeyboardObserver()

    def start(self) -> None:
        self.__gamepad.subscribe_callback_to_gripper(self.__gripper_callback)
        self.__gamepad.subscribe_callback_to_label(self.__label_callback)
        self.__gamepad.subscribe_callback_to_direction(self.__direction_callback)
        self.__gamepad.subscribe_callback_to_reset(self.__reset_callback)

        # self.__stream_outlet: pylsl.StreamOutlet = pylsl.StreamOutlet(pylsl.StreamInfo("Gamepad", "Gamepad",
        #                                                                                3, pylsl.cf_float32, "Gamepad"))

    def stop(self) -> None:
        self.__gamepad.unsubscribe_callback_to_gripper(self.__gripper_callback)
        self.__gamepad.unsubscribe_callback_to_label(self.__label_callback)
        self.__gamepad.unsubscribe_callback_to_direction(self.__direction_callback)

    def __gripper_callback(self, gripper: float) -> None:
        print("Gripper: " + gripper.__str__())

    def __label_callback(self, label: int) -> None:
        print("Label: " + label.__str__())

    def __direction_callback(self, direction: [float]) -> None:
        print("Direction: " + direction.__str__())

    def __reset_callback(self) -> None:
        print("Reset requested")




def main() -> None:
    logging.basicConfig()
    logging.root.setLevel(logging.NOTSET)

    gamepad_to_lsl: GamepadToLSL = GamepadToLSL()
    gamepad_to_lsl.start()

    while True:
        pass

if __name__ == '__main__':
    main()
