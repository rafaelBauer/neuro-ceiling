from tensordict import TensorDict
from tensordict.prototype import tensorclass
from torch import Tensor

from utils.sceneobservation import SceneObservation


@tensorclass
class ControllerStep:
    """
    The ControllerStep class is a tensorclass that represents a step in the controller.

    Attributes:
        action (Tensor): Represents the action taken in this step (a_t).
        scene_observation (SceneObservation): Represents the controller's observation of the scene/environment which lead to choose the action at this step (S_{t-1}).
        reward (Tensor): Represents the reward received which lead to choose the action at this step (R_{t-1}).
        episode_finished (Tensor): Represents whether the episode has finished at this step.
        extra_info (TensorDict): Represents any extra information at this step.
    """

    action: Tensor
    scene_observation: SceneObservation
    reward: Tensor
    episode_finished: Tensor        # For now this is not used
    extra_info: TensorDict
