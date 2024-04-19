import logging
import pylsl

from neuroceiling.configuration import DataStreamConfig
from neuroceiling.dataaquisition import IDataStream, DataStreamFactory

from neuroceiling import KeyboardObserver

gamepad: KeyboardObserver = KeyboardObserver()
eeg_data: dict[float, list[float]]


def main() -> None:
    logging.basicConfig()
    logging.root.setLevel(logging.NOTSET)

    datastream_config: DataStreamConfig = DataStreamConfig.load("run/Datastream.json")
    datastream: IDataStream = DataStreamFactory.get_datastream(datastream_config.datastream_config)

    # logging.info("Starting to record data to stop press X")

    datastream.setup_stream()
    # new data from datastream will be pushed via callback
    datastream.subscribe_to_new_data(new_eeg_data_callback)
    datastream.start_stream()

    while True:
        pass


def new_eeg_data_callback(timestamps: [float], new_data: [list]) -> None:
    global eeg_data
    # local_clock: float =  pylsl.local_clock()
    for timestamp, data in timestamps, new_data:
        eeg_data[timestamp] = data
        # print("Local clock: " + local_clock.__str__() + " Time LSL: " + timestamp.__str__() + " Data: " +
        #        new_data.__str__())


if __name__ == '__main__':
    main()
