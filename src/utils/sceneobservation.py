import torch
from tensordict import TensorDict, MemoryMappedTensor
from tensordict.prototype import tensorclass
from torch import Tensor


@tensorclass
class SceneObservation:
    _camera_observation: TensorDict
    _proprioceptive_obs: Tensor

    def __init__(self, camera_observation: dict, proprioceptive_obs: Tensor):
        self._camera_observation: TensorDict = TensorDict(camera_observation)
        self._proprioceptive_obs: Tensor = proprioceptive_obs

    @classmethod
    def empty(cls, batch_size=None, device=None):
        if batch_size is None:
            batch_size = []
        else:
            batch_size = [batch_size]
        data = cls(
            camera_observation=TensorDict({}, batch_size=batch_size, device=device),
            proprioceptive_obs=MemoryMappedTensor.empty(batch_size, dtype=torch.float, device=device),
            batch_size=batch_size,
            device=device,
        )
        return data

    @classmethod
    def from_list(cls, source_list: list["SceneObservation"], device=None):
        """
        Creates a SceneObservation object from a list of SceneObservation objects.

        Args:
            source_list (list[SceneObservation]): The list of SceneObservation objects.
            device (str, optional): The device on which the tensors should be allocated.

        Returns:
            SceneObservation: A SceneObservation object that contains the data from the source list.
        """
        data = cls(
            camera_observation=TensorDict({}, batch_size=[len(source_list)], device=device),
            proprioceptive_obs=MemoryMappedTensor.empty(
                (len(source_list), len(source_list[0].proprioceptive_obs.squeeze())), dtype=torch.float, device=device
            ),
            batch_size=[len(source_list)],
            device=device,
        )
        for i, scene_observation in enumerate(source_list):
            data[i] = cls(
                camera_observation=scene_observation.camera_observation,
                proprioceptive_obs=scene_observation.proprioceptive_obs.squeeze(),
                batch_size=[],
            )
        return data

    @property
    def camera_observation(self) -> TensorDict:
        return self._camera_observation

    @property
    def proprioceptive_obs(self) -> Tensor:
        return self._proprioceptive_obs

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
