from collect_demonstrations import Config

# from task.stack_cubes_a_config import config as task_config
# from task.stack_cubes_b_config import config as task_config
from task.stack_cubes_ind_spot_config import config as task_config


# ====== ManiSkill environment ========
from envs.maniskill import ManiSkillEnvironmentConfig

env_config = ManiSkillEnvironmentConfig(task_config=task_config)

# ====== Mock environment ========
# from envs.mock import MockEnvironmentConfig
#
# env_config = MockEnvironmentConfig()

# ====== Learn algorithm configuration ========
from learnalgorithm.ceilingalgorithm import CeilingAlgorithmConfig  # noqa
from learnalgorithm.learnalgorithm import NoLearnAlgorithmConfig  # noqa Nothing will happen with this learn algorithm

learn_algorithms = [
    NoLearnAlgorithmConfig(),
    NoLearnAlgorithmConfig(),
]

# ====== Controller configuration ========
from controller.periodiccontroller import PeriodicControllerConfig

# Controller in index 0 is the high level controller while in index N-1 is the low level controller.
# The controller at index N-1 will interact with the environment, while the others will interact with the controller
# at the next index.
controllers = [
    PeriodicControllerConfig(ACTION_TYPE="PickPlaceObject", polling_period_s=5, initial_goal=[0, 0, 0, 1]),
    PeriodicControllerConfig(
        ACTION_TYPE="TargetJointPositionAction", polling_period_s=0.05, initial_goal=[0, 0, 0, 0, 0, 0, 0]
    ),
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
    episodes=2,
    trajectory_size=45,
    task="StackCubesInd",
    feedback_type="pretrain_manual",
)
