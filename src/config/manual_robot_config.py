from manual_robot_control import Config

from task.stack_cubes_config import config as task_config

# ====== ManiSkill environment ========
from envs.maniskill import ManiSkillEnvironmentConfig

env_config = ManiSkillEnvironmentConfig(task_config=task_config)

# ====== Mock environment ========
# from envs.mock import MockEnvironmentConfig
#
# env_config = MockEnvironmentConfig()

# ====== Learn algorithm configuration ========
from learnalgorithm.learnalgorithm import LearnAlgorithmConfig

learn_algorithm_config = LearnAlgorithmConfig("DQN")

# ====== Policy configuration ========
from policy.manualobjectactionpolicy import ManualObjectActionPolicyConfig

high_level_policy_config = ManualObjectActionPolicyConfig()

from policy.motionplannerpolicy import MotionPlannerPolicyConfig

policy_config = MotionPlannerPolicyConfig()

# ====== Agent configuration ========
from controller.periodiccontroller import PeriodicControllerConfig

controller_config = PeriodicControllerConfig(polling_period_s=0.05, learn_algorithm_config=learn_algorithm_config)
high_level_controller_config = PeriodicControllerConfig(
    polling_period_s=5, learn_algorithm_config=learn_algorithm_config
)

config = Config(
    low_level_controller_config=controller_config,
    low_level_policy_config=policy_config,
    high_level_policy_config=high_level_policy_config,
    high_level_controller_config=high_level_controller_config,
    environment_config=env_config,
)
