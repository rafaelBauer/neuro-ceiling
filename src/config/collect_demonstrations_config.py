from collect_demonstrations import Config
from controller import ControllerBase
from policy import PolicyBase
from task.stack_cubes_config import config as task_config


# ====== ManiSkill environment ========
from envs.maniskill import ManiSkillEnvironmentConfig

env_config = ManiSkillEnvironmentConfig(task_config=task_config)

# ====== Mock environment ========
# from envs.mock import MockEnvironmentConfig
#
# env_config = MockEnvironmentConfig()

# ====== Learn algorithm configuration ========
from learnalgorithm.learnalgorithm import LearnAlgorithmBaseConfig

learn_algorithm_config = LearnAlgorithmBaseConfig("DQN")


# ====== Controller configuration ========
from controller.periodiccontroller import PeriodicControllerConfig

# Controller in index 0 is the high level controller while in index N-1 is the low level controller.
# The controller at index N-1 will interact with the environment, while the others will interact with the controller
# at the next index.
controllers = [
    PeriodicControllerConfig(polling_period_s=5, learn_algorithm_config=learn_algorithm_config),
    PeriodicControllerConfig(polling_period_s=0.05, learn_algorithm_config=learn_algorithm_config),
]

# ====== Policy configuration ========
from policy.manualobjectactionpolicy import ManualObjectActionPolicyConfig

manual_object_action_policy = ManualObjectActionPolicyConfig()

from policy.motionplannerpolicy import MotionPlannerPolicyConfig

motion_planner = MotionPlannerPolicyConfig()

# The policy at index 0 is added to controllers[0], the policy at index N-1 is added to controllers[N-1]
policies = [manual_object_action_policy, motion_planner]


config = Config(
    controllers=controllers,
    policies=policies,
    environment_config=env_config,
    episodes=2,
    trajectory_size=150,
)
