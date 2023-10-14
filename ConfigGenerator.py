from neuroceiling.configuration import NeuroCeilingConfig
from neuroceiling.dataaquisition import MoabbDatasetConfig


def generate_config():
    config = NeuroCeilingConfig("NeuroCeiling.json")

    config.dataset_config = MoabbDatasetConfig()
    config.dataset_config.dataset_name = "BNCI2014001"
    config.dataset_config.subject_ids = [1]

    print(config)
    config.save()


if __name__ == '__main__':
    generate_config()
