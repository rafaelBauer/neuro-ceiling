from __future__ import annotations

import abc
from typing import Optional

import braindecode.datasets


class DataStreamBaseConfig:
    """
    Base configuration class for every datastream.

    The configuration class tells the sotware which dataset is expected to be loaded.

    Every dataset configuration inherits from this class, and they extend it with the necessary
    information to make to be able to create its respective "implementation" object that inhertis from
    IDataStream.

    Attributes: __DATASET_TYPE: Constant that determines the dataset type. It must be given in the constructor by its
    child class.
    subject_ids: Specifies a list of subject(s) to be fetched. If None, data of all subjects is fetched.
    """

    def __init__(self, dataset_type: str):
        """
        Constructor.

        Parameters:
            dataset_type: Dataset type, meant to be written by child class
        """
        self.__DATASTREAM_TYPE: str = dataset_type
        self.subject_ids: list[int] = []

    @property
    def DATASTREAM_TYPE(self) -> str:
        return self.__DATASTREAM_TYPE


class IDataStream(metaclass=abc.ABCMeta):
    """ Interface class of a datastream.

    A datastream is meant to have a continuous stream of data. It could be live data as well as data from a Dataset file

    """

    def __init__(self, config: DataStreamBaseConfig):
        self.__TYPE_NAME: str = config.DATASTREAM_TYPE
        self._subject_ids: type(config.subject_ids) = config.subject_ids

    @abc.abstractmethod
    def setup_stream(self) -> bool:
        """
        Method sets up the stream according to the configuration. E.g. for the antNeuro cap, one needs to set up the
        LSL connection. If this stream is coming from a file, this step would most likely open the file, and so on.

        So this step is meant to do everything necessary to set up the stream and let everything be ready to when
        the start_stream method is called, the stream can be started.
        :return: If the setup was successful.
        """
        pass

    @abc.abstractmethod
    def start_stream(self) -> bool:
        """
        This will cause the start of the stream, and as new data arrives, it will call the callback function registered
        in subscribe_to_new_data from the context of the data arrival.

        :return: If the start of the stream was successful
        """
        pass

    @abc.abstractmethod
    def stop_stream(self) -> bool:
        pass

    @abc.abstractmethod
    def subscribe_to_new_data(self, callback_func) -> None:     # return subscription handle afterwards
        """

        :param callback_func: Function to be called when a new datapoint is available
        :return:
        """
        pass
    @property
    def TYPE_NAME(self) -> str:
        return self.__TYPE_NAME


class DataStreamFactory:
    @classmethod
    def get_datastream(cls, config: DataStreamBaseConfig) -> IDataStream:
        module = __import__("neuroceiling.dataaquisition." + str.lower(config.DATASTREAM_TYPE))
        class_ = getattr(getattr(getattr(module, "dataaquisition"), str.lower(config.DATASTREAM_TYPE)), config.DATASTREAM_TYPE)
        return class_(config)
