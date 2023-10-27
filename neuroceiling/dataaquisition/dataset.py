from __future__ import annotations

import abc
from typing import Optional

import braindecode.datasets


class DatasetBaseConfig:
    """
    Base configuration class for every dataset.

    The configuration class tells the software which dataset is expected to be loaded.

    Every dataset configuration inherits from this class, and they extend it with the necessary
    information to make to be able to create its respective "implementation" object that inhertis from
    IDataset.

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
        self.__DATASET_TYPE: str = dataset_type
        self.subject_ids: list[int] = []

    @property
    def DATASET_TYPE(self) -> str:
        return self.__DATASET_TYPE


class IDataset(metaclass=abc.ABCMeta):
    """ Interface class of a dataset.

        A dataset is a static, labeled bundle of data in which is used to train the brain decoding neural network.

        Every dataset implements this interface
    """

    def __init__(self, config: DatasetBaseConfig):
        self.__TYPE_NAME: str = config.DATASET_TYPE
        self.__subject_ids: type(config.subject_ids) = config.subject_ids
        self._raw_dataset: Optional[braindecode.datasets.BaseConcatDataset] = None

    @abc.abstractmethod
    def load_dataset(self) -> None:
        pass

    @property
    def TYPE_NAME(self) -> str:
        return self.__TYPE_NAME

    @property
    def subject_ids(self) -> list[int]:
        return self.__subject_ids

    @property
    def raw_dataset(self) -> Optional[braindecode.datasets.BaseConcatDataset]:
        return self._raw_dataset


class DatasetFactory:
    """
    Class that is responsible for creating the Dataset objects.
    """

    @classmethod
    def get_dataset(cls, config: DatasetBaseConfig) -> IDataset:
        """
        Static method meant to create an instance of the dataset based on its configuration.

        :param config: Configuration of dataset to be created
        :return: The dataset instance
        """
        module = __import__("neuroceiling.dataaquisition." + str.lower(config.DATASET_TYPE))
        class_ = getattr(getattr(getattr(module, "dataaquisition"), str.lower(config.DATASET_TYPE)),
                         config.DATASET_TYPE)
        return class_(config)
