from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from torch import Tensor


@dataclass
class RobotInfo:
    urdf_path: Path
    srdf_path: Path
    links: Sequence[str]
    joints: Sequence[str]
    end_effector_link: str

@dataclass
class RobotMotionInfo:
    current_qpos: Tensor
    current_ee_pose: Tensor
