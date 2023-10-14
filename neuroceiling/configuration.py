from __future__ import annotations
import jsonpickle
from typing import BinaryIO, TextIO

from .dataaquisition import DatasetBaseConfig


class NeuroCeilingConfig:
    """
    Test

    Attributes:
        filename: filename that will be stored
    """
    def __init__(self, filename: str):
        self.filename = filename
        self.dataset_config: DatasetBaseConfig = DatasetBaseConfig("")

    @classmethod
    def load(cls, filename) -> NeuroCeilingConfig:
        file: TextIO = open(filename)
        return jsonpickle.decode(file.read())

    def save(self):
        file: TextIO = open(self.filename, "w")
        json_object: str = jsonpickle.encode(self, indent=2)
        file.write(json_object)
