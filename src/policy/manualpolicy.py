from dataclasses import dataclass, field
from typing import Final, override

import torch
from torch import Tensor, cat

from policy.policy import PolicyBase, PolicyBaseConfig
from utils.keyboard_observer import KeyboardObserver
from utils.logging import log_constructor


@dataclass(kw_only=True)
class ManualPolicyConfig(PolicyBaseConfig):
    _POLICY_TYPE: str = field(init=False, default="ManualPolicy")


class ManualPolicy(PolicyBase):
    @log_constructor
    def __init__(self, config: ManualPolicyConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.__keyboard_observer: Final[KeyboardObserver] = KeyboardObserver()
        self.__keyboard_observer.start()

    @override
    def forward(self, states: Tensor) -> Tensor:
        # For when the keyboard observer is not working
        # action = torch.tensor(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dtype=torch.float32)
        # gripper = torch.tensor(np.array([0.0]), dtype=torch.float32)
        action = torch.tensor(self.__keyboard_observer.get_ee_action(), dtype=torch.float32)
        gripper = torch.tensor([self.__keyboard_observer.gripper], dtype=torch.float32)
        return cat([action, gripper])

    @override
    def update(self):
        pass
