import torch
from tensordict import TensorDict, MemoryMappedTensor
from tensordict.prototype import tensorclass
from torch import Tensor

from .gripperstate import GripperState


@tensorclass
class SceneObservation:
    """
    The SceneObservation class represents an observation of a scene.

    Attributes:
        _camera_observation (TensorDict): A dictionary of tensors representing the camera observation.
        _proprioceptive_obs (Tensor): A tensor representing the proprioceptive observation.
        _end_effector_pose (Tensor): A tensor representing the end effector pose.
        _objects (TensorDict): A dictionary of tensors representing the objects in the scene.
        _spots (TensorDict): A dictionary of tensors representing the spots in the scene.
    """

    _camera_observation: TensorDict
    _proprioceptive_obs: Tensor
    _end_effector_pose: Tensor
    _objects: TensorDict
    _spots: TensorDict

    def __init__(
        self,
        camera_observation: dict,
        proprioceptive_obs: Tensor,
        end_effector_pose: Tensor,
        objects: dict,
        spots: dict,
    ):
        """
        The constructor for the SceneObservation class.

        Args:
            camera_observation (dict): A dictionary representing the camera observation.
            proprioceptive_obs (Tensor): A tensor representing the proprioceptive observation.
            end_effector_pose (Tensor): A tensor representing the end effector pose.
            objects (dict): A dictionary representing the objects in the scene.
            spots (dict): A dictionary representing the spots in the scene.
        """
        self._camera_observation: TensorDict = TensorDict(camera_observation)
        self._proprioceptive_obs: Tensor = proprioceptive_obs
        self._end_effector_pose: Tensor = end_effector_pose
        self._objects: TensorDict = TensorDict(objects)
        self._spots: TensorDict = TensorDict(spots)

    @classmethod
    def empty(cls, batch_size=None, device=None):
        """
        Creates an empty SceneObservation object.

        Args:
            batch_size (int, optional): The batch size for the tensors. Defaults to None.
            device (str, optional): The device on which the tensors should be allocated. Defaults to None.

        Returns:
            SceneObservation: An empty SceneObservation object.
        """
        if batch_size is None:
            batch_size = []
        else:
            batch_size = [batch_size]
        data = cls(
            camera_observation=TensorDict({}, batch_size=batch_size, device=device),
            proprioceptive_obs=MemoryMappedTensor.empty(batch_size, dtype=torch.float, device=device),
            end_effector_pose=MemoryMappedTensor.empty(batch_size, dtype=torch.float, device=device),
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
            end_effector_pose=MemoryMappedTensor.empty(
                (len(source_list), len(source_list[0].end_effector_pose.squeeze())), dtype=torch.float, device=device
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
                end_effector_pose=scene_observation.end_effector_pose.squeeze(),
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
    def end_effector_pose(self) -> Tensor:
        return self._end_effector_pose

    @property
    def objects(self) -> TensorDict:
        return self._objects

    @property
    def spots(self) -> TensorDict:
        return self._spots

    @property
    def gripper_state(self) -> GripperState:
        if self._proprioceptive_obs.size(0) < 0:
            return GripperState.OPENED

        # 0.03 is the "closed" threshold for the gripper.
        if self._proprioceptive_obs.squeeze()[-1] > 0.03:
            return GripperState.OPENED
        return GripperState.CLOSED
