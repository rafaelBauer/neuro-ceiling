import importlib
from typing import Optional

from controller.controller import ControllerConfig, ControllerBase
from envs import BaseEnvironment
from learnalgorithm import LearnAlgorithm
from policy import PolicyBase

import goal, envs as actions


def create_controller(
    config: ControllerConfig,
    environment: BaseEnvironment,
    policy: PolicyBase,
    child_controller: Optional[ControllerBase] = None,
    learn_algorithm: Optional[LearnAlgorithm] = None,
) -> ControllerBase:
    config_module = importlib.import_module("." + str.lower(config.controller_type), "controller")
    controller_class = getattr(config_module, config.controller_type)
    try:
        # Get the action type from the actions module if it exists, otherwise from the goal module
        # Had to use the exception, since didn't find a better way to do this
        action_type = eval("actions." + config.ACTION_TYPE)
    except AttributeError:
        action_type = eval("goal." + config.ACTION_TYPE)
    return controller_class(config, environment, policy, action_type, child_controller, learn_algorithm)
