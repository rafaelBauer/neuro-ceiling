from __future__ import annotations

import threading

from typing import Optional, Callable, Any
import logging

from .datastream import IDataStream, DataStreamBaseConfig

from pylsl import StreamInlet, resolve_stream, StreamInfo


class AntNeuroCapDataStreamConfig(DataStreamBaseConfig):
    """ Configuration for streaming data from ant neuro cap.

        For more information about this dataset, read AntNeuroCapStreamData.
        It takes
        The name of the stream
        and the name of the file
        Attributes:
            hostname: The hostname from which the stream of data is being generated from
            stream_name: The name of the stream
            store_to_file_name: The name of the file to be stored. If this is "", the stream will not be stored.
    """

    def __init__(self):
        super().__init__("AntNeuroCapDataStream")
        self.hostname: str = ""
        self.stream_name: str = ""


class AntNeuroCapDataStream(IDataStream):
    """
    This type of data allows the user to stream the data from the AntNeuro cap and gives the possibility
    to have the stream saved to a file.

    """

    __REQUESTS_TIMEOUT_S: float = 10  # Timeout of the requests in seconds

    def __init__(self, config: AntNeuroCapDataStreamConfig):
        super().__init__(config)
        # Constants that should not be changed during the lifetime of the object
        self.__STREAM_NAME = config.stream_name
        self.__HOSTNAME = config.hostname

        self.__stream_inlet: Optional[StreamInlet] = None
        self.__current_stream_info: Optional[StreamInfo] = None

        self.__polling_thread: Optional[threading.Thread] = None
        self.__is_polling_stream: threading.Event = threading.Event()
        self.__callbacks: [Callable[[tuple[list, float, list[float]]], None]] = []

    def setup_stream(self) -> bool:
        """
        Sets up the stream, so we can, in a future step, start the stream and receive data.

        In order to set up the stream, the stream name must have been specified, otherwise there could be
        multiple streams available, and we would not know which stream to get the data from.

        In case the stream name is not defined, this method will print to the user the available streams as well as
        their names, so the user can define the correct name and open the stream.
        :return: if the open was successful
        """
        logging.info("Loading dataset: %s" % self.TYPE_NAME)

        streams: list[StreamInfo]
        # Verify if the stream name was defined, so we know which stream we can open.
        if self.__STREAM_NAME == "":
            logging.warning("Will not open any stream, but display available EEG streams."
                            "Please select one of the following streams by defining"
                            "its name (stream_name) in the configuration.")
            if self.__HOSTNAME == "":
                logging.info("Resolving stream by type: EEG")
                streams = resolve_stream('type', 'EEG')
            else:
                logging.info("Resolving stream by hostname: %s" % self.__HOSTNAME)
                streams = resolve_stream('hostname', self.__HOSTNAME)
        else:
            logging.info("Resolving stream by name: %s" % self.__STREAM_NAME)
            streams = resolve_stream('name', self.__STREAM_NAME)

        # Printing so we can see what can be opened.
        logging.info("Found the following streams:")
        logging.info("--------------------------------")
        stream: StreamInfo
        for index, stream in enumerate(streams):
            logging.info("  ID: %i" % index)
            logging.info("  Name: %s" % stream.name())
            logging.info("  Hostname: %s" % stream.hostname())
            logging.info("  Type: %s" % stream.type())
            logging.info("  Version: %s" % stream.version())
            logging.info("  Channel Count: %s" % stream.channel_count())
            logging.info("  Channel Format: %s" % stream.channel_format())
            logging.info("  Sampling rate: %s" % stream.nominal_srate())
            logging.info("--------------------------------")

        if len(streams) > 1:
            logging.error("There are more than one stream. Please specify only one to be opened.")
            return False
        self.__current_stream_info = streams[0]
        return True

    def subscribe_to_new_data(self, callback_func: Callable[[tuple[list, float, list[float]]], None]) -> None:
        # TODO: Make possible to unsubscribe from it.. maybe a subscription handle.
        self.__callbacks.append(callback_func)

    def start_stream(self) -> bool:
        """
        Opens the stream, so we start to receive data

        To open the stream its name must have been specified, otherwise there could be
        multiple streams available, and we would not know which stream to get the data from.

        In case the stream name is not defined, this method will print to the user the available streams as well as
        their names, so the user can define the correct name and open the stream.
        :return: If the open was successful
        """
        # This should never happen, but in case it does, we first close the previously opened stream
        self.stop_stream()

        # create a new inlet to read from the stream
        self.__stream_inlet = StreamInlet(self.__current_stream_info)
        logging.info("Opening LSL stream %s in hostname %s" % (
        self.__current_stream_info.name(), self.__current_stream_info.hostname()))
        self.__stream_inlet.open_stream(self.__REQUESTS_TIMEOUT_S)
        logging.info("LSL Stream successfully opened")

        self.__is_polling_stream.set()
        self.__polling_thread = threading.Thread(target=self.__poll_stream)
        self.__polling_thread.start()
        return True

    def __poll_stream(self) -> None:
        while self.__is_polling_stream.is_set():
            if self.__stream_inlet.samples_available() > 0:
                new_sample, timestamp = self.__stream_inlet.pull_sample(self.__REQUESTS_TIMEOUT_S)
                if new_sample:
                    for callback in self.__callbacks:
                        callback(timestamp, new_sample)
            self.__is_polling_stream.wait(0.01)  # waits for 10 ms.

    def stop_stream(self) -> bool:
        if self.__stream_inlet:
            current_stream_info: StreamInfo = self.__stream_inlet.info(self.__REQUESTS_TIMEOUT_S)
            logging.debug("Closing opened stream %s" % current_stream_info.name())
            self.__stream_inlet.close_stream()

        if self.__polling_thread:
            self.__is_polling_stream.clear()
            self.__polling_thread.join()

        return True
