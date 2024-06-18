from pre_train import Config
from task.stack_cubes_config import config as task_config


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
from learnalgorithm.learnalgorithm import LearnAlgorithmConfig

learn_algorithms = [
    BehaviorCloningAlgorithmConfig(batch_size=16, learning_rate=3e-4, weight_decay=3e-6, number_of_epochs=800),
    LearnAlgorithmConfig(
        batch_size=16, learning_rate=3e-4, weight_decay=3e-6
    ),  # Must have one. But it won't do nothing.
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
from policy.ceilingpolicy import CeilingPolicyConfig
from policy.manualrobotactionpolicy import ManualRobotActionPolicyConfig
from policy.motionplannerpolicy import MotionPlannerPolicyConfig

policy0 = CeilingPolicyConfig(
    visual_embedding_dim=256,
    proprioceptive_dim=9,
    action_dim=7,
    # from_file="/home/bauerr/git/neuro-ceiling/data/StackCubesA/pretrain_manual_policy.pt"
)
# policy0 = ManualObjectActionPolicyConfig()
policy1 = MotionPlannerPolicyConfig()

# The policy at index 0 is added to controllers[0], the policy at index N-1 is added to controllers[N-1]
policies = [policy0, policy1]


config = Config(
    controllers=controllers,
    policies=policies,
    learn_algorithms=learn_algorithms,
    environment_config=env_config,
    episodes=100,
    task="StackCubesA",
    feedback_type="ceiling_full",  # ceiling_full, pretrain_manual
    dataset_name="demos_2.dat",
)