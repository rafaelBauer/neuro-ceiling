import logging
from typing import Any

from neuroceiling.configuration import DataStreamConfig
from neuroceiling.dataaquisition import IDataStream, DataStreamFactory

from neuroceiling import KeyboardObserver

gamepad: KeyboardObserver = KeyboardObserver()


def main() -> None:
    logging.basicConfig()
    logging.root.setLevel(logging.NOTSET)

    datastream_config: DataStreamConfig = DataStreamConfig.load("run/Datastream.json")
    datastream_config: IDataStream = DataStreamFactory.get_datastream(datastream_config.datastream_config)

    datastream_config.setup_stream()
    # new data from datastream will be pushed via callback
    datastream_config.subscribe_to_new_data(new_data_callback)
    datastream_config.start_stream()

    while True:
        pass


def new_data_callback(new_data: tuple[list, Any]) -> None:
    print("New data received: " + new_data.__str__())
    gamepad.get_label()


if __name__ == '__main__':
    main()
