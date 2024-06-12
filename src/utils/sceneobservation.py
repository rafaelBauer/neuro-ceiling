import torch
from tensordict import TensorDict
from tensordict.prototype import tensorclass
from torch import Tensor

# class CameraObservation:
#     _name: str
#     _observation: Tensor


@tensorclass
class SceneObservation:
    _camera_observation: TensorDict
    _proprioceptive_obs: Tensor

    def __init__(self, camera_observation: dict, proprioceptive_obs: Tensor):
        self._camera_observation: TensorDict = TensorDict(camera_observation)
        self._proprioceptive_obs: Tensor = proprioceptive_obs

    @property
    def observation(self) -> TensorDict:
        return self._observation

    # @abstractmethod
    # def get_raw_observation(self) -> torch.Tensor:
    #     pass
    #
    # @abstractmethod
    # def to_pose(self) -> Pose:
    #     pass
    #
    # @abstractmethod
    # def to_joint_position(self) -> JointPosition:
    #     pass