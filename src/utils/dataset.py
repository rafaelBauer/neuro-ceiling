import random

import numpy
import torch
from tensordict import TensorDict, MemoryMappedTensor
from tensordict.prototype import tensorclass
from torch import Tensor
from torch.utils.data import Dataset

from utils.human_feedback import HumanFeedback
from utils.sceneobservation import SceneObservation


@tensorclass
class TrajectoryData:
    """
    The TrajectoryData class is a tensorclass that represents the data of a trajectory.

    Attributes:
        scene_observation (SceneObservation): Represents the observation of the scene in this trajectory.
        action (Tensor): Represents the action taken in this trajectory.
        feedback (Tensor): Represents the feedback received in this trajectory.
        object_poses (TensorDict): Represents the poses of the objects in this trajectory.
        spots (TensorDict): Represents the spots in this trajectory.
    """

    scene_observation: SceneObservation
    action: Tensor
    feedback: Tensor

    @classmethod
    def empty(cls, device=None):
        """
        Creates an empty TrajectoryData object.

        Args:
            device (str, optional): The device on which the tensors should be allocated.

        Returns:
            TrajectoryData: An empty TrajectoryData object.
        """
        data = cls(
            scene_observation=SceneObservation.empty(device=device),
            action=torch.tensor([], device=device),
            feedback=torch.tensor([], device=device),
            batch_size=[],
            device=device,
        )
        return data

    @classmethod
    def from_list(cls, source_list: list["TrajectoryData"], device=None):
        """
        Creates a TrajectoryData object from a list of TrajectoryData objects.

        Args:
            source_list (list[TrajectoryData]): The list of TrajectoryData objects.
            device (str, optional): The device on which the tensors should be allocated.

        Returns:
            TrajectoryData: A TrajectoryData object that contains the data from the source list.
        """
        scene_observation_list = [trajectory_data.scene_observation for trajectory_data in source_list]
        data = cls(
            scene_observation=SceneObservation.from_list(scene_observation_list, device=device),
            action=MemoryMappedTensor.empty(
                (len(source_list), len(source_list[0].action)), dtype=torch.float, device=device
            ),
            feedback=MemoryMappedTensor.empty(
                (len(source_list), len(source_list[0].feedback)), dtype=torch.float, device=device
            ),
            batch_size=[len(source_list)],
            device=device,
        )
        for i, trajectory_data in enumerate(source_list):
            data[i] = cls(
                scene_observation=trajectory_data.scene_observation,
                action=trajectory_data.action,
                feedback=trajectory_data.feedback,
                batch_size=[],
            )
        return data


class TrajectoriesDataset(Dataset):
    """
    The TrajectoriesDataset class is a subclass of PyTorch's Dataset class. It holds a collection of trajectories.

    Attributes:
        __trajectory_size (int): The size of each trajectory.
        __current_trajectory (list[TrajectoryData]): The current trajectory being built.
        __trajectories (TrajectoryData): The collection of trajectories (batch_size, num_steps_per_trajectory).
        __good_count (int): The count of good feedback in current trajectory.
        __corrected_count (int): The count of corrected feedback in current trajectory.
        __bad_count (int): The count of bad feedback in current trajectory.
    """

    def __init__(self, trajectory_size: int):
        self.__trajectory_size = trajectory_size
        self.__current_trajectory: list[TrajectoryData] = []
        self.__trajectories: TrajectoryData = TrajectoryData.empty()
        self.__good_count = 0
        self.__corrected_count = 0
        self.__bad_count = 0
        return

    def __getitem__(self, idx):
        """
        Retrieves a copy of a trajectory at the desired index from the dataset.

        Args:
            idx (int): The index of the trajectory to retrieve.

        Returns:
            TrajectoryData: The retrieved trajectory.
        """
        if self.__corrected_count < 10:
            alpha = 1
        else:
            alpha = (self.__good_count + self.__bad_count) / self.__corrected_count

        trajectory = self.__trajectories[idx].copy()
        for trajectory_step in trajectory:
            if trajectory_step.feedback == HumanFeedback.CORRECTED:
                trajectory_step.feedback = alpha

        return trajectory

    def __len__(self) -> int:
        """
        Returns the number of trajectories in the dataset.

        Returns:
            int: The number of trajectories.
        """
        return len(self.__trajectories)

    def add(self, step: TrajectoryData):
        """
        Adds a step to the current trajectory and updates the feedback counts.

        Args:
           step (TrajectoryData): The step to add.
        """
        self.__current_trajectory.append(step)

        assert isinstance(step.feedback, Tensor), f"Expected feedback to be a tensor, got {type(step.feedback)}"

        if step.feedback[0] == HumanFeedback.GOOD:
            self.__good_count += 1
        elif step.feedback[0] == HumanFeedback.CORRECTED:
            self.__corrected_count += 1
        elif step.feedback[0] == HumanFeedback.BAD:
            self.__bad_count += 1
        return

    def save_current_traj(self):
        """
        Saves the current trajectory to the collection of trajectories and resets the current trajectory.
        """
        self.__current_trajectory = self.__down_sample_current_trajectory()
        current_trajectory = TrajectoryData.from_list(self.__current_trajectory)
        # If it is empty, then it is the first trajectory
        if self.__trajectories.batch_size == torch.Size([]):
            self.__trajectories = current_trajectory.unsqueeze(0)
        else:
            self.__trajectories = torch.cat([self.__trajectories, current_trajectory.unsqueeze(0)])
        self.reset_current_traj()
        return

    def reset_current_traj(self):
        """
        Resets the current trajectory.
        """
        self.__current_trajectory = []
        return

    def sample(self, batch_size) -> TrajectoryData:
        """
        Samples a batch of trajectories from the dataset.

        Args:
            batch_size (int): The size of the batch to sample.

        Returns:
            A TrajectoryData object with the desired batch size (batch_size, num_steps_per_trajectory).
        """
        batch_size = min(batch_size, len(self))
        indices = random.sample(range(len(self)), batch_size)
        return torch.stack([*[self[i] for i in indices]], dim=1)

    def __down_sample_current_trajectory(self):
        """
        Down-samples the current trajectory to match the trajectory size.

        Returns:
            list[TrajectoryData]: The down-sampled trajectory.
        """
        if len(self.__current_trajectory) == self.__trajectory_size:
            return self.__current_trajectory
        if len(self.__current_trajectory) < self.__trajectory_size:
            return self.__current_trajectory + [self.__current_trajectory[-1]] * (
                self.__trajectory_size - len(self.__current_trajectory)
            )

        indices = numpy.linspace(start=0, stop=len(self.__current_trajectory) - 1, num=self.__trajectory_size)
        indices = numpy.round(indices).astype(int)
        return numpy.array([self.__current_trajectory[i] for i in indices])
