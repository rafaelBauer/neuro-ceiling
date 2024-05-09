import importlib

from agent.agent import AgentConfig, AgentBase


def create_agent(config: AgentConfig) -> AgentBase:
    config_module = importlib.import_module("agent", "agent")
    agent_class = getattr(config_module, "AgentBase")
    return agent_class(config)
