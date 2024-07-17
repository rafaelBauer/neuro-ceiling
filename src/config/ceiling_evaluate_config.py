from pre_train import Config

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
from learnalgorithm.behaviorcloningalgorithm import BehaviorCloningAlgorithmConfig
from learnalgorithm.ceilingalgorithm import CeilingAlgorithmConfig
from learnalgorithm.learnalgorithm import LearnAlgorithmConfig, NoLearnAlgorithmConfig

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
    PeriodicControllerConfig(ACTION_TYPE="PickPlaceObject", polling_period_s=5, initial_goal=[0, 0, 0, 1], max_steps=30, log_metrics=True),
    PeriodicControllerConfig(
        ACTION_TYPE="TargetJointPositionAction", polling_period_s=0.05, initial_goal=[0, 0, 0, 0, 0, 0, 0]
    ),
]

# ====== Policy configuration ========
from policy.manualobjectactionpolicy import ManualObjectActionPolicyConfig
from policy.ceilingpolicy import CeilingPolicyConfig
from policy.manualrobotactionpolicy import ManualRobotActionPolicyConfig
from policy.motionplannerpolicy import MotionPlannerPolicyConfig

# The policy at index 0 is added to controllers[0], the policy at index N-1 is added to controllers[N-1]
policies = [
    CeilingPolicyConfig(
        visual_embedding_dim=256,
        proprioceptive_dim=9,
        action_dim=4,
        from_file="ceiling_trained_policy_new.pt",
    ),
    MotionPlannerPolicyConfig(),
]


config = Config(
    controllers=controllers,
    policies=policies,
    learn_algorithms=learn_algorithms,
    environment_config=env_config,
    episodes=100,
    task="StackCubesInd",
)
