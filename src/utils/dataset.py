import random

import numpy
import torch
from tensordict import TensorDict
from torch import Tensor
from torch.utils.data import Dataset

from utils.human_feedback import HumanFeedback


class TrajectoryData:
    """
    Data structure that contains the data for a single step in a trajectory.
    """

    camera_obs: TensorDict
    proprioceptive_obs: Tensor
    action: Tensor
    feedback: Tensor
    object_poses: TensorDict
    spots: TensorDict


class TrajectoriesDataset(Dataset):
    """
    Dataset that holds a collection of trajectories.
    """

    def __init__(self, trajectory_size: int):
        self.__trajectory_size = trajectory_size
        self.__current_trajectory: list[TrajectoryData] = []
        self.__trajectories: list[list[TrajectoryData]] = [[]]
        self.__good_count = 0
        self.__corrected_count = 0
        self.__bad_count = 0
        return

    def __getitem__(self, idx):
        if self.__corrected_count < 10:
            alpha = 1
        else:
            alpha = (self.__good_count + self.__bad_count) / self.__corrected_count

        trajectory = self.__trajectories[idx].copy()
        for trajectory_step in trajectory:
            if trajectory_step.feedback == HumanFeedback.CORRECTED:
                trajectory_step.feedback = alpha

        return trajectory

    def __len__(self):
        return len(self.__trajectories)

    def add(self, step: TrajectoryData):
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
        self.__current_trajectory = self.__down_sample_current_trajectory()
        self.__trajectories.append(self.__current_trajectory)
        self.reset_current_traj()
        return

    def reset_current_traj(self):
        self.__current_trajectory = []
        return

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self))
        indices = random.sample(range(len(self)), batch_size)
        batch = zip(*[self[i] for i in indices])
        camera_batch = torch.stack(next(batch), dim=1)
        proprio_batch = torch.stack(next(batch), dim=1)
        action_batch = torch.stack(next(batch), dim=1)
        feedback_batch = torch.stack(next(batch), dim=1)
        return camera_batch, proprio_batch, action_batch, feedback_batch

    def __down_sample_current_trajectory(self):
        if len(self.__current_trajectory) == self.__trajectory_size:
            return self.__current_trajectory
        if len(self.__current_trajectory) < self.__trajectory_size:
            return self.__current_trajectory + [self.__current_trajectory[-1]] * (
                self.__trajectory_size - len(self.__current_trajectory)
            )

        indices = numpy.linspace(start=0, stop=len(self.__current_trajectory) - 1, num=self.__trajectory_size)
        indices = numpy.round(indices).astype(int)
        return numpy.array([self.__current_trajectory[i] for i in indices])
