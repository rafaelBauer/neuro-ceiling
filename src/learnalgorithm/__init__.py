import importlib
from .learnalgorithm import LearnAlgorithm, LearnAlgorithmConfig


def create_learn_algorithm(config: LearnAlgorithmConfig, **kwargs) -> LearnAlgorithm:
    if config.algo_type == "NoLearnAlgorithm":
        return None
    config_module = importlib.import_module("." + str.lower(config.algo_type), "learnalgorithm")
    learn_algorithm_class: LearnAlgorithm = getattr(config_module, config.algo_type)
    return learn_algorithm_class(config, **kwargs)
