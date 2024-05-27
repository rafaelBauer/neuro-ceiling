import importlib

from agent.agent import AgentConfig, AgentBase
from envs import BaseEnvironment
from policy import PolicyBase


def create_agent(config: AgentConfig, environment: BaseEnvironment, policy: PolicyBase) -> AgentBase:
    config_module = importlib.import_module("agent", "agent")
    agent_class = getattr(config_module, "AgentBase")
    return agent_class(config, environment, policy)
