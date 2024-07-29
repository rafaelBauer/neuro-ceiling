from pre_train import Config

# from task.stack_cubes_a_config import config as task_config
# from task.stack_cubes_b_config import config as task_config
from task.stack_cubes_ind_spot_config import config as task_config


# ====== ManiSkill environment ========
from envs.maniskill import ManiSkillEnvironmentConfig

env_config = ManiSkillEnvironmentConfig(task_config=task_config, headless=False)

# ====== Mock environment ========
# from envs.mock import MockEnvironmentConfig
#
# env_config = MockEnvironmentConfig()

# ====== Learn algorithm configuration ========
from learnalgorithm.behaviorcloningalgorithm import BehaviorCloningAlgorithmConfig
from learnalgorithm.ceilingalgorithm import CeilingAlgorithmConfig
from learnalgorithm.learnalgorithm import LearnAlgorithmConfig, NoLearnAlgorithmConfig

# from learnalgorithm.feedbackdevice.keyboardfeedback import KeyboardFeedbackConfig
from learnalgorithm.feedbackdevice.automaticfeedback import AutomaticFeedbackConfig

feedback_device_config = AutomaticFeedbackConfig(action_dim=4, task_config=task_config)
learn_algorithms = [
    CeilingAlgorithmConfig(
        batch_size=16,
        learning_rate=3e-4,
        weight_decay=3e-6,
        steps_per_episode=45,
        feedback_device_config=feedback_device_config,
        load_dataset="demos_10.dat",
        save_dataset="demos_ceiling.dat",  # Number of episodes will be appended to the name before the extension
    ),
    NoLearnAlgorithmConfig(),
]

# ====== Controller configuration ========
from controller.periodiccontroller import PeriodicControllerConfig

# Controller in index 0 is the high level controller while in index N-1 is the low level controller.
# The controller at index N-1 will interact with the environment, while the others will interact with the controller
# at the next index.
controllers = [
    PeriodicControllerConfig(
        ACTION_TYPE="PickPlaceObject", polling_period_s=5, initial_goal=[0, 0, 0, 1], log_metrics=True
    ),
    PeriodicControllerConfig(
        ACTION_TYPE="TargetJointPositionAction", polling_period_s=0.05, initial_goal=[0, 0, 0, 0, 0, 0, 0]
    ),
]

# ====== Policy configuration ========
from policy.ceilingpolicy import CeilingPolicyConfig
from policy.motionplannerpolicy import MotionPlannerPolicyConfig

# The policy at index 0 is added to controllers[0], the policy at index N-1 is added to controllers[N-1]
policies = [
    CeilingPolicyConfig(
        visual_embedding_dim=256,
        proprioceptive_dim=9,
        action_dim=feedback_device_config.action_dim,
        from_file="ceiling_10_pretrain_policy_0.pt",
        save_to_file=feedback_device_config.name
        + learn_algorithms[0].name
        + "_policy.pt",  # Number of episodes will be appended to the name before the extension
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
