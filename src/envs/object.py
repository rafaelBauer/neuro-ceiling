from typing import List, Final

from utils.pose import Pose


class Object:
    """
    The Object class represents a physical object with a pose and height.

    Attributes:
        __pose (Pose): The pose of the object.
        __height (float): The height of the object.
    """

    def __init__(self, init_pose: Pose, height: float = 0.02):
        """
        The constructor for the Object class.

        Parameters:
            init_pose (Pose): The initial pose of the object.
            height (float): The height of the object. Default is 0.02.
        """
        self.__pose = init_pose
        self.__height = height
        pass

    @property
    def pose(self) -> Pose:
        """
        The pose property of the object.

        Returns:
            Pose: The current pose of the object.
        """
        return self.__pose

    @pose.setter
    def pose(self, new_pose: Pose):
        """
        The pose setter of the object.

        Parameters:
            new_pose (Pose): The new pose of the object.
        """
        self.__pose = new_pose

    @property
    def height(self) -> float:
        """
        The height property of the object.

        Returns:
            float: The current height of the object.
        """
        return self.__height


class Spot:
    """
    The Spot class represents a location that can contain multiple objects.

    Attributes:
        _objects (List[Object]): The list of objects at this spot.
        _pose (Final[Pose]): The pose of the spot.
    """

    def __init__(self, **kwargs):
        """
        The constructor for the Spot class.

        Parameters:
            kwargs: Variable length argument list. Can contain "pose" or "object".
        """
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
        """
        The pose property of the spot.

        Returns:
            Pose: The current pose of the spot.
        """
        return self._pose

    def add_object(self, obj: Object):
        """
        Adds an object to the spot.

        Parameters:
            obj (Object): The object to be added.
        """
        if obj not in self._objects:
            self._objects.append(obj)

    def remove_object(self, obj: Object):
        """
        Removes an object from the spot.

        Parameters:
            obj (Object): The object to be removed.
        """
        if obj in self._objects:
            self._objects.remove(obj)
