from __future__ import annotations

from braindecode.datasets import MOABBDataset

from .dataset import IDataset, DatasetBaseConfig
import logging


class MoabbDatasetConfig(DatasetBaseConfig):
    """ Configuration of a MOABB Dataset
        http://moabb.neurotechx.com/docs/datasets.html

        This configuration object is used by
    """

    def __init__(self):
        super().__init__("MoabbDataset")
        self.dataset_name: str = ""


class MoabbDataset(IDataset):
    def __init__(self, config: MoabbDatasetConfig):
        super().__init__(config)
        self.__name = config.dataset_name

    def load_dataset(self) -> None:
        logging.info("Loading dataset: %s" % self.TYPE_NAME)
        self._raw_dataset = MOABBDataset(dataset_name=self.__name, subject_ids=self.subject_ids)

