from enum import IntEnum


class GripperState(IntEnum):
    """
    An enumeration representing the commands that can be sent to the gripper.

    Attributes
    ----------
    OPENED : int
       The gripper is opened. The value of it 1.
    CLOSED : int
       The gripper is closed. The value of it -1.
    """

    OPENED = 1
    CLOSED = -1
