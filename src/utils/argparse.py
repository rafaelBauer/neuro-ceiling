import argparse

from omegaconf import DictConfig, OmegaConf

from utils.config import build_dict_config_object


def resolve_tuple(*args):
    return tuple(args)


OmegaConf.register_new_resolver("as_tuple", resolve_tuple)


def parse_args(program_description: str, data_load: bool = False, need_task: bool = False,
               extra_args: tuple = tuple()) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=program_description)
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="Path to config file to use.",
    )

    if need_task:
        parser.add_argument(
            "-t",
            "--task",
            default="PhoneOnBase",
            type=str,
            help="Name of the task to train on.",
        )

    if data_load:
        parser.add_argument(
            "-f",
            "--feedback_type",
            default="pretrain_manual",
            type=str,
            help="The training data type. Cloning, dcm, ...",
        )
        parser.add_argument(
            "--path",
            default=None,
            help="Path to a dataset. May be provided instead of f-t.",
        )

    for arg in extra_args:
        arg_name = arg.pop("name")
        parser.add_argument(arg_name, **arg)

    parser.add_argument(
        "-o",
        "--overwrite",
        nargs="+",
        default=[],
        help="Overwrite config values. Format: key=value. Keys need to be "
             "fully qualified. E.g. "
             "'training.steps=1000 observation.cameras=overhead'",
    )
    __args__ = parser.parse_args()
    return __args__


def get_config_from_args(program_description: str, data_load: bool = False, need_task: bool = False,
                         extra_args: tuple = tuple()) -> tuple[argparse.Namespace, DictConfig]:
    args = parse_args(program_description, data_load, need_task, extra_args)
    config = build_dict_config_object(args.config, args.overwrite)

    return args, config