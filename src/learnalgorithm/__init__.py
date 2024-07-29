import importlib

# from utils.keyboard_observer import KeyboardObserver
from .feedbackdevice.feedbackdevice import FeedbackDevice, FeedbackDeviceConfig
from .learnalgorithm import LearnAlgorithm, LearnAlgorithmConfig


def create_learn_algorithm(config: LearnAlgorithmConfig, keyboard_observer, **kwargs) -> LearnAlgorithm:
    if config.algo_type == "NoLearnAlgorithm":
        return None
    config_module = importlib.import_module("." + str.lower(config.algo_type), "learnalgorithm")
    learn_algorithm_class: LearnAlgorithm = getattr(config_module, config.algo_type)
    if hasattr(config, "feedback_device_config"):
        feedback_device = create_feedback_device(config.feedback_device_config, keyboard_observer=keyboard_observer)
    else:
        feedback_device = None
    return learn_algorithm_class(config, feedback_device=feedback_device, **kwargs)


def create_feedback_device(config: FeedbackDeviceConfig, **kwargs) -> FeedbackDevice:
    config_module = importlib.import_module(".feedbackdevice." + str.lower(config.device_type), "learnalgorithm")
    feedback_device_class: FeedbackDevice = getattr(config_module, config.device_type)
    return feedback_device_class(config, **kwargs)
