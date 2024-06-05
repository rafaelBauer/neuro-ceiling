import importlib
from typing import Optional

from controller.controller import ControllerConfig, ControllerBase
from envs import BaseEnvironment
from policy import PolicyBase


def create_controller(
    config: ControllerConfig,
    environment: BaseEnvironment,
    policy: PolicyBase,
    child_controller: Optional[ControllerBase] = None,
) -> ControllerBase:
    config_module = importlib.import_module("." + str.lower(config.controller_type), "controller")
    agent_class = getattr(config_module, config.controller_type)
    return agent_class(config, environment, policy, child_controller)
