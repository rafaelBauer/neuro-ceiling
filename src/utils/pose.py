from enum import Enum

import mplib

import numpy

import transforms3d


class RotationRepresentation(Enum):
    """
    Enum for the different rotation representations
    """
    QUATERNION = "quaternion",
    EULER = "euler"


class Pose:
    """
    Pose that can be converted into different frames.
    Internally it is stored as a mplib.Pose
    """

    __pose: mplib.Pose

    def __getstate__(self) -> tuple:
        return self.__pose.__getstate__()

    def __imul__(self, other: 'Pose') -> 'Pose':
        """
        Overloading operator *= for ``Pose<S> *= Pose<S>``
        """
        self.__pose *= other.__pose
        return self

    def __init__(self, **kwargs) -> None:
        """
        Constructs a default Pose with p = (0,0,0) and q = (1,0,0,0)
        """
        self.__pose = mplib.Pose(**kwargs)
        self.__rotation_representation: dict[RotationRepresentation, numpy.ndarray] = {
            RotationRepresentation.QUATERNION: self.q,
            RotationRepresentation.EULER: self.euler
        }

    def __mul__(self, other: 'Pose') -> 'Pose':
        """
        Overloading operator * for ``Pose<S> * Pose<S>``
        """
        return Pose(obj=(self.__pose * other.__pose))

    def __repr__(self) -> str:
        return self.__pose.__repr__()

    def __setstate__(self, arg0: tuple) -> None:
        self.__pose.__setstate__(arg0)

    @property
    def p(self) -> numpy.ndarray:
        return self.__pose.p

    @p.setter
    def p(self, value):
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
        theta, omega = transforms3d.quaternions.quat2axangle(self.__pose.q)
        return_val = transforms3d.euler.axangle2euler(theta, omega)
        return numpy.array(return_val)

    @property
    def raw_quaternion(self) -> numpy.ndarray:
        return numpy.hstack([self.p, self.__rotation_representation[RotationRepresentation.QUATERNION]])

    @property
    def raw_euler(self) -> numpy.ndarray:
        return numpy.hstack([self.p, self.__rotation_representation[RotationRepresentation.EULER]])

    @property
    def mplib_pose(self) -> mplib.Pose:
        return self.__pose

    def get_raw_pose(self, rotation_representation: RotationRepresentation):
        return numpy.hstack([self.p, self.__rotation_representation[rotation_representation]])

    def inv(self):
        return self.__pose.inv()
