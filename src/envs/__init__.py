import importlib
from .environment import BaseEnvironmentConfig, BaseEnvironment

# class Environment(Enum):
#     PANDA = "panda"
#     MANISKILL = "maniskill"
#
#
# def get_env(env_str):
#     return Environment[env_str.upper()]


# class EnvironmentFactory:
#     """
#     Class that is responsible for creating the Environment objects.
#     """
#
#     @classmethod
#     def get_environment(cls, config: DatasetBaseConfig) -> IDataset:
#         """
#         Static method meant to create an instance of the dataset based on its configuration.
#
#         :param config: Configuration of dataset to be created
#         :return: The dataset instance
#         """
#         module = __import__("neuroceiling.dataaquisition." + str.lower(config.DATASET_TYPE))
#         class_ = getattr(getattr(getattr(module, "dataaquisition"), str.lower(config.DATASET_TYPE)),
#                          config.DATASET_TYPE)
#         return class_(config)

def create_environment(config: BaseEnvironmentConfig) -> BaseEnvironment:
    # env_type = config.env
    config_module = importlib.import_module('.' + str.lower(config.env_type), "envs")
    env_class = getattr(config_module, config.env_type + 'Env')
    return env_class(config)
    # if env_type is Environment.PANDA:
    #     from env.franka import FrankaEnv as Env
    # elif env_type is Environment.MANISKILL:
    #     from env.mani_skill import ManiSkillEnv as Env
    # else:
    #     raise ValueError("Invalid environment {}".format(config.env))
    #
    # return Env
