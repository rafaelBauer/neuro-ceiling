from manual_robot_control import Config

# ====== ManiSkill environment ========
from envs.maniskill import ManiSkillEnvironmentConfig

env_config = ManiSkillEnvironmentConfig()


# ====== Mock environment ========
# from envs.mock import MockEnvironmentConfig
#
# env_config = MockEnvironmentConfig()

# ====== Learn algorithm configuration ========
from learnalgorithm.learnalgorithm import LearnAlgorithmBaseConfig

learn_algorithm_config = LearnAlgorithmBaseConfig('DQN')

# ====== Policy configuration ========
from policy.manualpolicy import ManualPolicyConfig

policy_config = ManualPolicyConfig()

# ====== Agent configuration ========
from agent.agent import AgentConfig

agent_config = AgentConfig(
    polling_period_s=0.05,
    learn_algorithm_config=learn_algorithm_config,
    policy_config=policy_config,
    environment=env_config
)

config = Config(agent_config=agent_config)
