from neuroceiling.configuration import NeuroCeilingConfig, DatasetConfig, DataStreamConfig
from neuroceiling.dataaquisition import MoabbDatasetConfig
from neuroceiling.dataaquisition import AntNeuroCapDataStreamConfig, XdfFileConfig

def generate_config():
    datasetConfig = DatasetConfig("run/DatasetConfig.json")

    datasetConfig.dataset_config = XdfFileConfig()
    datasetConfig.dataset_config.filename = "run/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg.xdf"
    # datasetConfig.dataset_config.subject_ids = [1]

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
