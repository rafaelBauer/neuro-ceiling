import torch
from tensordict import TensorDict, MemoryMappedTensor
from tensordict.prototype import tensorclass
from torch import Tensor


@tensorclass
class SceneObservation:
    _camera_observation: TensorDict
    _proprioceptive_obs: Tensor
    _objects: TensorDict
    _spots: TensorDict

    def __init__(self, camera_observation: dict, proprioceptive_obs: Tensor, objects: dict, spots: dict):
        self._camera_observation: TensorDict = TensorDict(camera_observation)
        self._proprioceptive_obs: Tensor = proprioceptive_obs
        self._objects: TensorDict = TensorDict(objects)
        self._spots: TensorDict = TensorDict(spots)

    @classmethod
    def empty(cls, batch_size=None, device=None):
        if batch_size is None:
            batch_size = []
        else:
            batch_size = [batch_size]
        data = cls(
            camera_observation=TensorDict({}, batch_size=batch_size, device=device),
            proprioceptive_obs=MemoryMappedTensor.empty(batch_size, dtype=torch.float, device=device),
            objects=TensorDict({}, batch_size=batch_size, device=device),
            spots=TensorDict({}, batch_size=batch_size, device=device),
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
            objects=TensorDict({}, batch_size=[len(source_list)], device=device),
            spots=TensorDict({}, batch_size=[len(source_list)], device=device),
            batch_size=[len(source_list)],
            device=device,
        )
        for i, scene_observation in enumerate(source_list):
            data[i] = cls(
                camera_observation=scene_observation.camera_observation,
                proprioceptive_obs=scene_observation.proprioceptive_obs.squeeze(),
                objects=scene_observation.objects,
                spots=scene_observation.spots,
                batch_size=[],
            )
        return data

    @property
    def camera_observation(self) -> TensorDict:
        return self._camera_observation

    @property
    def proprioceptive_obs(self) -> Tensor:
        return self._proprioceptive_obs

    @property
    def objects(self) -> TensorDict:
        return self._objects

    @property
    def spots(self) -> TensorDict:
        return self._spots
