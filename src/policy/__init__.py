import importlib
from .policy import PolicyBaseConfig, PolicyBase


def create_policy(config: PolicyBaseConfig) -> PolicyBase:
    config_module = importlib.import_module('.' + str.lower(config.policy_type), "policy")
    policy_class: PolicyBase = getattr(config_module, config.policy_type)
    return policy_class(config)

