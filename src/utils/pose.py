import copy
from enum import Enum

import mplib

import numpy
import torch

import transforms3d


class RotationRepresentation(Enum):
    """
    Enum for the different rotation representations
    """

    QUATERNION = ("quaternion",)
    EULER = "euler"


class Pose:
    """
    Pose that can be converted into different frames.
    Internally it is stored as a mplib.Pose
    """

    __pose: mplib.Pose

    def __getstate__(self) -> tuple:
        """
        Get the state of the Pose object. This is used for serialization.
        """
        return self.__pose.__getstate__()

    def __imul__(self, other: "Pose") -> "Pose":
        """
        Overloading operator *= for ``Pose<S> *= Pose<S>``
        This allows for in-place multiplication of Pose objects.
        """
        self.__pose *= other.__pose
        return self

    def __init__(self, **kwargs) -> None:
        """
        Constructs a default Pose with p = (0,0,0) and q = (1,0,0,0)
        """
        if "euler" in kwargs:
            kwargs["q"] = transforms3d.euler.euler2quat(kwargs["euler"][0], kwargs["euler"][1], kwargs["euler"][2])
            del kwargs["euler"]
        self.__pose = mplib.Pose(**kwargs)
        self.__rotation_representation: dict[RotationRepresentation, numpy.ndarray] = {
            RotationRepresentation.QUATERNION: self.q,
            RotationRepresentation.EULER: self.euler,
        }

    def __mul__(self, other: "Pose") -> "Pose":
        """
        Overloading operator * for ``Pose<S> * Pose<S>``
        This allows for multiplication of Pose objects.
        """
        return Pose(obj=(self.__pose * other.__pose))

    def __repr__(self) -> str:
        """
        Returns a string representation of the Pose object.
        """
        return self.__pose.__repr__()

    def __setstate__(self, arg0: tuple) -> None:
        """
        Set the state of the Pose object. This is used for deserialization.
        """
        self.__init__(p=arg0[:3], q=arg0[3:])

    def __eq__(self, other):
        return numpy.all(self.__pose.p == other.__pose.p) and numpy.all(self.__pose.q == other.__pose.q)

    @property
    def p(self) -> numpy.ndarray:
        """
        Returns the position (x,y,z) of the Pose object.
        """
        return self.__pose.p

    @p.setter
    def p(self, value):
        """
        Sets the position (x,y,z) of the Pose object.
        """
        self.__pose.p = value

    @property
    def q(self) -> numpy.ndarray:
        """
        Rotation representation in quaternion (w, x, y, z)
        """
        return self.__pose.q

    @property
    def euler(self) -> numpy.ndarray:
        """
        Rotation representation in euler angles, AKA axis angles, (x, y, z)
        """
        vector, theta = transforms3d.quaternions.quat2axangle(self.__pose.q)
        return_val = transforms3d.euler.axangle2euler(vector, theta)
        return numpy.array(return_val)

    @property
    def raw_quaternion(self) -> numpy.ndarray:
        """
        Returns the raw quaternion representation of the Pose object.
        """
        return numpy.hstack([self.p, self.__rotation_representation[RotationRepresentation.QUATERNION]])

    @property
    def raw_euler(self) -> numpy.ndarray:
        """
        Returns the raw euler representation of the Pose object.
        """
        return numpy.hstack([self.p, self.__rotation_representation[RotationRepresentation.EULER]])

    @property
    def mplib_pose(self) -> mplib.Pose:
        """
        Returns the underlying mplib.Pose object.
        """
        return self.__pose

    def get_raw_pose(self, rotation_representation: RotationRepresentation):
        """
        Returns the raw pose representation of the Pose object based on the provided rotation representation.
        """
        return numpy.hstack([self.p, self.__rotation_representation[rotation_representation]])

    def inv(self):
        """
        Returns the inverse of the Pose object.
        """
        return self.__pose.inv()

    def copy(self):
        return copy.copy(self)

    def to_tensor(self, rotation_representation: RotationRepresentation):
        return torch.from_numpy(self.get_raw_pose(rotation_representation))
