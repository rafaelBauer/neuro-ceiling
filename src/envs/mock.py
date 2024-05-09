from dataclasses import dataclass
from typing import override

import numpy as np

from envs import BaseEnvironment, BaseEnvironmentConfig
from utils.logging import logger


@dataclass
class MockEnvironmentConfig(BaseEnvironmentConfig):
    def __init__(self):
        super().__init__("Mock")


class MockEnv(BaseEnvironment):
    def __init__(self, config: BaseEnvironmentConfig) -> None:
        super().__init__(config)

    @override
    def reset_joint_pose(self) -> None:
        logger.info("Reset joint poses")

    @override
    def close(self) -> None:
        logger.info("Closed Environment")

    @override
    def start(self) -> None:
        logger.info("Started Environment")

    @override
    def _step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        logger.info("Stepped with action: {}", action)
        test: tuple[dict, float, bool, dict] = {"test": 1}, 0.5, True, {"test2": 2}
        return test

    @override
    def reset(self, **kwargs) -> None:
        logger.info("Reset environment")
