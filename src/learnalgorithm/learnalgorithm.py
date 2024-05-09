from dataclasses import dataclass, field
from typing import Final


@dataclass
class LearnAlgorithmBaseConfig:
    __ALGO_TYPE: str = field(init=True)


class LearnAlgorithmBase:
    def __init__(self, config: LearnAlgorithmBaseConfig):
        pass
