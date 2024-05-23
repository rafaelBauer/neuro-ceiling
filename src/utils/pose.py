import mplib

import numpy
from scipy.spatial.transform import Rotation

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

    @property
    def q(self) -> numpy.ndarray:
        return self.__pose.q

    @property
    def euler(self) -> numpy.ndarray:
        return Rotation.from_quat(self.__pose.q).as_euler('xyz', degrees=True)

    @property
    def raw_quartenion(self) -> numpy.ndarray:
        return numpy.hstack([self.p, self.q])

    @property
    def raw_euler(self) -> numpy.ndarray:
        return numpy.hstack([self.p, self.euler])

    @property
    def mplib_pose(self) -> mplib.Pose:
        return self.__pose
