from dataclasses import dataclass, field
from typing import override

from torch import Tensor
from mplib import Planner

from policy.policy import PolicyBaseConfig, PolicyBase
from utils.logging import logger, log_constructor


@dataclass(kw_only=True)
class MotionPlannerPolicyConfig(PolicyBaseConfig):
    _POLICY_TYPE: str = field(init=False, default="MotionPlannerPolicy")


class MotionPlannerPolicy(PolicyBase):
    @log_constructor
    def __init__(self, config: MotionPlannerPolicyConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config

    @override
    def update(self):
        pass

    @override
    def forward(self, states: Tensor) -> Tensor:
        pass
