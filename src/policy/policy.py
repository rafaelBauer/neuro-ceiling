from dataclasses import dataclass
from typing import Final

from torch import nn

from envs import BaseEnvironment


class PolicyBase(nn.Module):
    def __init__(self, environment: BaseEnvironment):
        super(PolicyBase, self).__init__()
        self.environment: Final[BaseEnvironment] = environment

    def update(self):
        pass

    def sample_action(self, state):
        pass
