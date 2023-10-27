from neuroceiling.configuration import NeuroCeilingConfig, DatasetConfig, DataStreamConfig
from neuroceiling.dataaquisition import MoabbDatasetConfig
from neuroceiling.dataaquisition import AntNeuroCapDataStreamConfig

def generate_config():
    datasetConfig = DatasetConfig("run/DatasetConfig.json")

    datasetConfig.dataset_config = MoabbDatasetConfig()
    datasetConfig.dataset_config.dataset_name = "BNCI2014001"
    datasetConfig.dataset_config.subject_ids = [1]

    datasetConfig.save()

    datastreamconfig = DataStreamConfig("run/Datastream.json")

    datastreamconfig.datastream_config = AntNeuroCapDataStreamConfig()
    datastreamconfig.datastream_config.hostname = ""
    datastreamconfig.datastream_config.stream_name = ""
    datastreamconfig.datastream_config.subject_ids = [1]

    datastreamconfig.save()

    neuroCeilingConfig = NeuroCeilingConfig("run/NeuroCeiling.json")

    neuroCeilingConfig.save()



if __name__ == '__main__':
    generate_config()
