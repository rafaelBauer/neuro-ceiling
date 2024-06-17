from collect_demonstrations import Config
from task.stack_cubes_config import config as task_config


# ====== ManiSkill environment ========
from envs.maniskill import ManiSkillEnvironmentConfig

env_config = ManiSkillEnvironmentConfig(task_config=task_config)

# ====== Mock environment ========
# from envs.mock import MockEnvironmentConfig
#
# env_config = MockEnvironmentConfig()

# ====== Learn algorithm configuration ========
from learnalgorithm.ceilingalgorithm import CeilingAlgorithmConfig  # noqa
from learnalgorithm.learnalgorithm import LearnAlgorithmConfig  # noqa Nothing will happen with this learn algorithm

learn_algorithms = [
    # TODO: Create one learn algorithm to use to collect demonstrations
    LearnAlgorithmConfig(batch_size=16, learning_rate=3e-4, weight_decay=3e-6),
    LearnAlgorithmConfig(batch_size=16, learning_rate=3e-4, weight_decay=3e-6),
]

# ====== Controller configuration ========
from controller.periodiccontroller import PeriodicControllerConfig

# Controller in index 0 is the high level controller while in index N-1 is the low level controller.
# The controller at index N-1 will interact with the environment, while the others will interact with the controller
# at the next index.
controllers = [
    PeriodicControllerConfig(polling_period_s=5),
    PeriodicControllerConfig(polling_period_s=0.05),
]

# ====== Policy configuration ========
from policy.manualobjectactionpolicy import ManualObjectActionPolicyConfig
from policy.manualrobotactionpolicy import ManualRobotActionPolicyConfig
from policy.motionplannerpolicy import MotionPlannerPolicyConfig

policy0 = ManualObjectActionPolicyConfig()
policy1 = MotionPlannerPolicyConfig()

# The policy at index 0 is added to controllers[0], the policy at index N-1 is added to controllers[N-1]
policies = [
    ManualObjectActionPolicyConfig(),
    MotionPlannerPolicyConfig(),
]


config = Config(
    controllers=controllers,
    policies=policies,
    learn_algorithms=learn_algorithms,
    environment_config=env_config,
    episodes=10,
    trajectory_size=200,
    task="StackCubesA",
    feedback_type="pretrain_manual",
)
