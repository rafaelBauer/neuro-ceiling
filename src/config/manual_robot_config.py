from manual_robot_control import Config
from envs.maniskill import ManiSkillEnvironmentConfig

maniskill_config = ManiSkillEnvironmentConfig()


config = Config(env_config=maniskill_config)
