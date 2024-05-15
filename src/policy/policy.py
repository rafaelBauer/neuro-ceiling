from abc import abstractmethod
from dataclasses import dataclass, field

from torch import nn, Tensor


@dataclass(kw_only=True)
class PolicyBaseConfig:
    _POLICY_TYPE: str = field(init=True)

    @property
    def policy_type(self) -> str:
        return self._POLICY_TYPE


class PolicyBase(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def forward(self, states: Tensor) -> Tensor:
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary (differentiable) operations on Tensors.
        """
        pass
