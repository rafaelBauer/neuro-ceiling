from typing import List

from envs.object import Spot, Object


class Scene:
    def __init__(self, objects: List[Object], spots: List[Spot]):
        self.__objects = objects
        self.__spots = spots

    @property
    def objects(self):
        return self.__objects

    @property
    def spots(self):
        return self.__spots
