import numpy as np

from .keyboard_observer import HumanFeedback


def human_feedback(keyboard_obs, action, feedback_type) -> tuple[HumanFeedback, np.array]:
    if feedback_type == "evaluative":
        feedback = keyboard_obs.get_label()

    elif feedback_type == "dagger":
        if keyboard_obs.is_direction_commanded or keyboard_obs.is_gripper_commanded:
            action = correct_action(keyboard_obs, action)
            feedback = HumanFeedback.CORRECTED
        else:
            feedback = HumanFeedback.BAD

    elif feedback_type == "iwr":
        if keyboard_obs.is_direction_commanded or keyboard_obs.is_gripper_commanded:
            action = correct_action(keyboard_obs, action)
            feedback = HumanFeedback.CORRECTED
        else:
            feedback = HumanFeedback.GOOD

    elif feedback_type == "ceiling_full":
        if keyboard_obs.is_direction_commanded or keyboard_obs.is_gripper_commanded:
            action = correct_action(keyboard_obs, action)
            feedback = HumanFeedback.CORRECTED
        else:
            feedback = keyboard_obs.get_label()

    elif feedback_type == "ceiling_partial":
        if keyboard_obs.is_direction_commanded or keyboard_obs.is_gripper_commanded:
            action = correct_action(keyboard_obs, action, full_control=False)
            feedback = HumanFeedback.CORRECTED
        else:
            feedback = keyboard_obs.get_label()

    else:
        raise NotImplementedError("Feedback type not supported!")
    return action, feedback


def correct_action(keyboard_obs, action, full_control=True) -> np.array:
    if full_control:
        action[:-1] = keyboard_obs.get_ee_action()
    elif keyboard_obs.is_direction_commanded:
        ee_step = keyboard_obs.get_ee_action()
        action[:-1] = action[:-1] * 0.5 + ee_step
        action = np.clip(action, -0.9, 0.9)
    if keyboard_obs.is_gripper_commanded:
        action[-1] = keyboard_obs.get_gripper()
    return action
