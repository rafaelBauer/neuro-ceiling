from __future__ import annotations

import abc
from typing import Optional

import braindecode.datasets


class DatasetBaseConfig:
    """
    Base configuration class for every dataset.

    The configuration class tells the sotware which dataset is expected to be loaded.

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
    """ Base configuration class for every dataset.


    """

    def __init__(self, config: DatasetBaseConfig):
        self.__TYPE_NAME: str = config.DATASET_TYPE
        self._subject_ids: type(config.subject_ids) = config.subject_ids
        self._raw_dataset: Optional[braindecode.datasets.BaseConcatDataset] = None

    @classmethod
    def get_dataset(cls, config: DatasetBaseConfig) -> IDataset:
        module = __import__("neuroceiling.dataaquisition." + str.lower(config.DATASET_TYPE))
        class_ = getattr(getattr(getattr(module, "dataaquisition"), str.lower(config.DATASET_TYPE)), config.DATASET_TYPE)
        return class_(config)

    @abc.abstractmethod
    def load_dataset(self) -> None:
        pass

    @property
    def TYPE_NAME(self) -> str:
        return self.__TYPE_NAME

    @property
    def raw_dataset(self) -> Optional[braindecode.datasets.BaseConcatDataset]:
        return self._raw_dataset
