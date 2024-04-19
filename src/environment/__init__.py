from enum import Enum


class Environment(Enum):
    PANDA = "panda"
    MANISKILL = "maniskill"


def get_env(env_str):
    return Environment[env_str.upper()]


def import_env(config):
    env_type = config.env
    if env_type is Environment.PANDA:
        from env.franka import FrankaEnv as Env
    elif env_type is Environment.MANISKILL:
        from env.mani_skill import ManiSkillEnv as Env
    else:
        raise ValueError("Invalid environment {}".format(config.env))

    return Env
