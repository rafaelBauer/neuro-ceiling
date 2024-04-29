import importlib
import os
import sys
from pathlib import Path
from typing import Final

from omegaconf import DictConfig, OmegaConf


def import_config_file(config_file: Path):
    """
    Import a config file as a python module.

    Parameters
    ----------
    config_file : str or path
        Path to the config file.

    Returns
    -------
    module
        Python module containing the config file.

    """
    config_file = str(config_file)
    if config_file[-3:] == ".py":
        config_file = config_file[:-3]

    config_file_path = os.path.abspath("/".join(config_file.split("/")[:-1]))
    sys.path.insert(1, config_file_path)
    file: Final[str] = config_file.rsplit("/", maxsplit=1)[-1]
    config = importlib.import_module(file, package=None)

    return config


def build_dict_config_object(config_file: Path, overwritten_config: list[str]) -> DictConfig:
    conf_file = import_config_file(config_file)
    config = conf_file.config
    config = OmegaConf.structured(config)

    overwrites = OmegaConf.from_dotlist(overwritten_config)
    config = OmegaConf.merge(config, overwrites)

    assert isinstance(config, DictConfig)

    return config


class ConfigBase:
    def __init__(self):
        pass

    def to_yaml(self) -> str:
        return OmegaConf.to_yaml(self, resolve=True)
