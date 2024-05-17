from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class RobotInfo:
    urdf_path: Path
    srdf_path: Path
    links: Sequence[str]
    joints: Sequence[str]
    end_effector_link: str

@dataclass
class RobotMotionInfo:
    current_end_effector_pose: Sequence[float]
