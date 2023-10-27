from __future__ import annotations
import jsonpickle
from typing import BinaryIO, TextIO

from .dataaquisition import DatasetBaseConfig, DataStreamBaseConfig


class BaseConfig:
    def __init__(self, filename: str):
        self.filename = filename

    @classmethod
    def load(cls, filename) -> BaseConfig:
        file: TextIO = open(filename)
        return jsonpickle.decode(file.read())

    def save(self) -> None:
        file: TextIO = open(self.filename, "w")
        json_object: str = jsonpickle.encode(self, indent=2)
        file.write(json_object)


class NeuroCeilingConfig(BaseConfig):
    """

    """

    def __init__(self, filename: str):
        super().__init__(filename)


class DatasetConfig(BaseConfig):
    """

    """

    def __init__(self, filename: str):
        super().__init__(filename)
        self.dataset_config: DatasetBaseConfig = DatasetBaseConfig("")


class DataStreamConfig(BaseConfig):
    """

    """

    def __init__(self, filename: str):
        super().__init__(filename)
        self.datastream_config: DataStreamBaseConfig = DataStreamBaseConfig("")
