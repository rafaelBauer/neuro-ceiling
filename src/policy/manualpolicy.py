from dataclasses import dataclass, field
from typing import Final, override

import torch
from torch import Tensor, cat

from policy.policy import PolicyBase, PolicyBaseConfig
from utils.keyboard_observer import KeyboardObserver
from utils.logging import log_constructor


@dataclass(kw_only=True)
class ManualPolicyConfig(PolicyBaseConfig):
    """
    Configuration class for ManualPolicy. Inherits from PolicyBaseConfig.
    """

    _POLICY_TYPE: str = field(init=False, default="ManualPolicy")


class ManualPolicy(PolicyBase):
    """
    ManualPolicy class that inherits from PolicyBase. This class is used to manually control the policy.
    """

    @log_constructor
    def __init__(self, config: ManualPolicyConfig, keyboard_observer: KeyboardObserver, **kwargs):
        """
        Constructor for the ManualPolicy class.

        This constructor initializes the ManualPolicy object with the provided configuration and keyword arguments.

        :param config: Configuration object for the ManualPolicy. This should be an instance of ManualPolicyConfig.
        :param kwargs: Additional keyword arguments.
        """
        super().__init__(config, **kwargs)
        self.__keyboard_observer: Final[KeyboardObserver] = keyboard_observer

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
        """
        Update method for the ManualPolicy class. This method is currently not implemented.
        """
        pass