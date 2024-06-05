from dataclasses import dataclass, field
from typing import Final


@dataclass
class LearnAlgorithmBaseConfig:
    __ALGO_TYPE: str = field(init=True)

    @property
    def algo_type(self) -> str:
        return self.__ALGO_TYPE


class LearnAlgorithmBase:
    def __init__(self, config: LearnAlgorithmBaseConfig):
        pass
