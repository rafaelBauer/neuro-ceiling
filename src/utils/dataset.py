import random

import torch
from tensordict import tensorclass, TensorDict, MemoryMappedTensor
from torch import Tensor
from torch.utils.data import Dataset

from utils.human_feedback import HumanFeedback


@tensorclass
class Observation:
    camera_obs: TensorDict
    proprioceptive_obs: Tensor  # Joint positions, etc
    action: Tensor
    feedback: Tensor
    object_poses: TensorDict
    spots: TensorDict

    # @classmethod
    # def from_dataset(cls, dataset, device=None):
    #     data = cls(
    #         camera_obs=MemoryMappedTensor.empty((len(dataset), *dataset[0][0].squeeze().shape), dtype=torch.float32),
    #         proprioceptive_obs=MemoryMappedTensor.empty((len(dataset),), dtype=torch.int64),
    #         action=MemoryMappedTensor.empty((len(dataset),), dtype=torch.float32),
    #         feedback=MemoryMappedTensor.empty((len(dataset),), dtype=torch.int),
    #         object_poses=MemoryMappedTensor.empty((len(dataset),), dtype=torch.int64),
    #         spots=MemoryMappedTensor.empty((len(dataset),), dtype=torch.int64),
    #         batch_size=[len(dataset)],
    #         device=device,
    #     )
    #     for i, (image, target) in enumerate(dataset):
    #         data[i] = cls(camera_obs=camera_obs, proprioceptive_obs=torch.tensor(target), batch_size=[])
    #     return data


class Trajectory:
    def __init__(self, batch_size: int):
        self.__observation: list[Observation] = []
        self.__batch_size: int = batch_size
        self.__feedback_good_count = 0
        self.__feedback_corrected_count = 0
        self.__feedback_bad_count = 0

    def add_observation(self, observation: Observation):
        self.__observation.append(observation)

        if observation.feedback[0] == HumanFeedback.GOOD:
            self.__feedback_good_count += 1
        elif observation.feedback[0] == HumanFeedback.CORRECTED:
            self.__feedback_corrected_count += 1
        elif observation.feedback[0] == HumanFeedback.BAD:
            self.__feedback_bad_count += 1
        # return

    def __len__(self):
        return len(self.__observation)

    def __getitem__(self, idx):
        return self.__observation[idx]

    @property
    def observations(self) -> list[Observation]:
        return self.__observation

    @property
    def batch_size(self):
        return self.__batch_size

    @property
    def feedback_good_count(self):
        return self.__feedback_good_count

    @property
    def feedback_corrected_count(self):
        return self.__feedback_corrected_count

    @property
    def feedback_bad_count(self):
        return self.__feedback_bad_count


class TrajectoriesDataset(Dataset):
    def __init__(self, batch_size: int):
        self.__trajectories: list[Trajectory] = []
        self.__batch_size: int = batch_size

    def __getitem__(self, idx) -> Observation:
        if self.__trajectories[idx].feedback_corrected_count < 10:
            alpha = 1
        else:
            alpha = ((self.__trajectories[idx].feedback_good_count + self.__trajectories[idx].feedback_bad_count) /
                     self.__trajectories[idx].feedback_corrected_count)
        weighted_feedback = [
            alpha if observation.feedback == HumanFeedback.CORRECTED else observation.feedback
            for observation in self.__trajectories[idx].observations
        ]
        weighted_feedback = torch.tensor(weighted_feedback).unsqueeze(1)
        return Observation(camera_obs=self.__trajectories[idx].observations.camera_obs)

    # camera_obs = self.__trajectories[idx].observations.camera_obs,
    # proprioceptive_obs = self.__trajectories[idx].proprio_obs,
    # action = self.__trajectories[idx].action,
    # feedback = weighted_feedback,
    # object_poses = self.__trajectories[idx].observations.object_poses,
    # spots = self.__trajectories[idx].observations.spots,
    # batch_size = [self.__trajectories[idx].batch_size]

    def __len__(self):
        return len(self.__trajectories)

    def save_trajectory(self, new_trajectory: Trajectory):
        self.__trajectories.append(new_trajectory)

    # def save_current_traj(self):
    #     camera_obs = downsample_traj(self.current_camera_obs, self.sequence_len)
    #     proprio_obs = downsample_traj(self.current_proprio_obs, self.sequence_len)
    #     action = downsample_traj(self.current_action, self.sequence_len)
    #     feedback = downsample_traj(self.current_feedback, self.sequence_len)
    #     camera_obs_th = torch.tensor(camera_obs, dtype=torch.float32)
    #     proprio_obs_th = torch.tensor(proprio_obs, dtype=torch.float32)
    #     action_th = torch.tensor(action, dtype=torch.float32)
    #     feedback_th = torch.tensor(feedback, dtype=torch.float32)
    #     self.camera_obs.append(camera_obs_th)
    #     self.proprio_obs.append(proprio_obs_th)
    #     self.action.append(action_th)
    #     self.feedback.append(feedback_th)
    #     self.reset_current_traj()
    #     return

    # def reset_current_traj(self):
    #     self.current_camera_obs = []
    #     self.current_proprio_obs = []
    #     self.current_action = []
    #     self.current_feedback = []
    #     return
    #
    # def sample(self, batch_size):
    #     batch_size = min(batch_size, len(self))
    #     indeces = random.sample(range(len(self)), batch_size)
    #     batch = zip(*[self[i] for i in indeces])
    #     camera_batch = torch.stack(next(batch), dim=1)
    #     proprio_batch = torch.stack(next(batch), dim=1)
    #     action_batch = torch.stack(next(batch), dim=1)
    #     feedback_batch = torch.stack(next(batch), dim=1)
    #     return camera_batch, proprio_batch, action_batch, feedback_batch
