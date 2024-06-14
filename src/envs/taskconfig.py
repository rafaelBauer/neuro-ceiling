from dataclasses import dataclass
from typing import Any

from envs.object import Object
from utils.pose import Pose


@dataclass
class TaskConfig:
    """
    Configuration class for the task.
    """

    _initial_objects: dict[str, dict[str, Any]]
    _available_spots_pose: dict[str, list[float]]
    _target_objects_pose: dict[str, list[float]]

    @property
    def initial_objects(self) -> dict[str, Object]:
        initial_objects: dict[str, Object] = {}
        for key, value in self._initial_objects.items():
            kwargs = value.copy()
            pose = kwargs["pose"]
            del kwargs["pose"]
            initial_objects[key] = Object(init_pose=Pose(p=pose[:3], q=pose[3:]), **kwargs)
        return initial_objects

    @property
    def available_spots_pose(self) -> dict[str, Pose]:
        available_spots_pose: dict[str, Pose] = {}
        for key, value in self._available_spots_pose.items():
            available_spots_pose[key] = Pose(p=value[:3], q=value[3:])
        return available_spots_pose

    @property
    def target_objects_pose(self) -> dict[str, Pose]:
        target_objects_pose: dict[str, Pose] = {}
        for key, value in self._target_objects_pose.items():
            target_objects_pose[key] = Pose(p=value[:3], q=value[3:])
        return target_objects_pose
