import copy
from dataclasses import dataclass, field
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

    # only used for serialization
    __pose: mplib.Pose = field(init=True)

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
        if "raw_pose" in kwargs:
            kwargs["p"] = kwargs["raw_pose"][:3]
            kwargs["q"] = kwargs["raw_pose"][3:]
            del kwargs["raw_pose"]
        if "raw_euler_pose" in kwargs:
            kwargs["p"] = kwargs["raw_euler_pose"][:3]
            kwargs["euler"] = kwargs["raw_euler_pose"][3:]
            del kwargs["raw_euler_pose"]

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
        if not isinstance(other, Pose):
            return False
        return numpy.all(self.__pose.p == other.__pose.p) and transforms3d.quaternions.nearly_equivalent(self.__pose.q, other.__pose.q)

    def is_close(self, other, atol=1e-8) -> bool:
        if not isinstance(other, Pose):
            return False
        return numpy.allclose(self.__pose.p, other.__pose.p, atol=atol) and transforms3d.quaternions.nearly_equivalent(
            self.__pose.q, other.__pose.q, atol=0.1
        )

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
        """
        Creates a copy of the Pose object.

        Returns:
            Pose: A copy of the Pose object.
        """
        return copy.copy(self)

    def to_tensor(self, rotation_representation: RotationRepresentation):
        """
        Converts the Pose object to a tensor.

        Args:
            rotation_representation (RotationRepresentation): The rotation representation to use for the conversion.

        Returns:
            torch.Tensor: A tensor representing the Pose object.
        """
        return torch.from_numpy(self.get_raw_pose(rotation_representation))

    def is_same_xy_position(self, other, atol=1e-8):
        """
        Checks if the X and Y positions of the Pose object are the same as those of another Pose object.

        Args:
            other (Pose): The other Pose object to compare with.
            atol (float, optional): The absolute tolerance parameter for the numpy allclose function. Defaults to 1e-8.

        Returns:
            bool: True if the X and Y positions are the same, False otherwise.
        """
        return numpy.allclose(self.__pose.p[:2], other.__pose.p[:2], atol=atol)
