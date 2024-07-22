import torch
import seaborn as sns  #
import matplotlib.pyplot as plt

from utils.dataset import TrajectoriesDataset

dataset_name = "data/StackCubesInd/demos_ceiling_50.dat"


def main():
    replay_buffer: TrajectoriesDataset = torch.load(dataset_name)
    bins = {}
    for trajectory in replay_buffer:
        # Sort by initial object position
        # objects = sorted(trajectory.scene_observation.objects.items(), key=lambda y: y[1][0][1])
        colors = {"Cube A": "Cube R", "Cube B": "Cube G", "Cube C": "Cube B"}
        dic = {colors[k]: v for k, v in trajectory.scene_observation.objects.items()}
        objects = sorted(dic.items(), key=lambda y: y[1][0][1])
        initial_object_positions = ", ".join([name for name, pos in objects])
        bins[initial_object_positions] = bins.get(initial_object_positions, 0) + 1

    data = {"initial_positions": bins}
    # sns.displot(data, kind="hist")
    plt.bar(bins.keys(), bins.values())
    plt.xticks(rotation=90)
    plt.xlabel("Initial object positions")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
