from typing import List, Final

from utils.pose import Pose


class Object:
    def __init__(self, init_pose: Pose, height: float = 0.02):
        self.__pose = init_pose
        self.__height = height
        pass

    @property
    def pose(self) -> Pose:
        return self.__pose

    @pose.setter
    def pose(self, new_pose: Pose):
        self.__pose = new_pose

    @property
    def height(self) -> float:
        return self.__height


class Spot:
    def __init__(self, **kwargs):
        self._objects: List[Object] = []

        if "pose" in kwargs:
            init_pose: Pose = kwargs["init_pose"]
        elif "object" in kwargs:
            obj: Object = kwargs["object"]
            init_pose: Pose = obj.pose
            self.add_object(obj)
        else:
            raise ValueError("pose or object must be provided")

        self._pose: Final[Pose] = init_pose

    @property
    def pose(self) -> Pose:
        return self._pose

    def add_object(self, obj: Object):
        if obj not in self._objects:
            self._objects.append(obj)

    def remove_object(self, obj: Object):
        if obj in self._objects:
            self._objects.remove(obj)
