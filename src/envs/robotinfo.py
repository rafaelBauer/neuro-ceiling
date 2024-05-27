from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from torch import Tensor

from utils.pose import Pose


@dataclass
class RobotInfo:
    """
    A dataclass representing the information about a robot.

    Attributes
    ----------
    urdf_path : Path
        The path to the URDF file of the robot.
    srdf_path : Path
        The path to the SRDF file of the robot.
    links : Sequence[str]
        A sequence of strings representing the names of the links of the robot.
    joints : Sequence[str]
        A sequence of strings representing the names of the joints of the robot.
    end_effector_link : str
        The name of the end effector link of the robot.
    """

    urdf_path: Path
    srdf_path: Path
    links: Sequence[str]
    joints: Sequence[str]
    end_effector_link: str


@dataclass
class RobotMotionInfo:
    """
    A dataclass representing the motion information of a robot.

    Attributes
    ----------
    current_qpos : Tensor
        The current joint positions of the robot as a tensor.
    current_ee_pose : Tensor
        The current pose of the end effector of the robot as a tensor.
    """

    current_qpos: Tensor
    current_ee_pose: Pose
