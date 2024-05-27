import importlib
from .environment import BaseEnvironmentConfig, BaseEnvironment


def create_environment(config: BaseEnvironmentConfig) -> BaseEnvironment:
    config_module = importlib.import_module("." + str.lower(config.env_type), "envs")
    env_class = getattr(config_module, config.env_type + "Env")
    return env_class(config)
