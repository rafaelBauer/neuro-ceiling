import sys
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

from tqdm import tqdm
from loguru import logger

logger_indentation = ContextVar("logger_indentation", default=0)


# Context manager for indentation taken from https://github.com/Delgan/loguru/issues/424#issuecomment-826074098  # noqa: E501
@contextmanager
def indent_logs(indent_size: int = 2):
    val = logger_indentation.get()
    logger_indentation.set(val + indent_size)
    yield
    logger_indentation.set(val)


#
#
# def patcher(msg):
#     indentation = logger_indentation.get()
#     return " " * indentation + msg


class DuplicateFilter:
    def __init__(self):
        self.msgs = set()

    def __call__(self, record):
        unseen = (msg := record["message"]) not in self.msgs
        self.msgs.add(msg)
        return unseen or not record["extra"].get("filter", True)


def write_tqdm(msg):
    tqdm.write(msg, end="")


def formatter(record):
    indentation = 0
    indentation = " " * indentation
    form = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> |"
        " <level>{level: <8}</level> | " + indentation + " <level>{message}</level>\n"
    )
    return form


def setup_logger() -> logger:
    logger.remove()
    logger.add(write_tqdm, colorize=True, format=formatter, level="DEBUG")

    return logger


def log_constructor(init_func):
    """
    Simple decorator to log the initialization of a class. Logs the class name
    that is initialized and indents all logs that are created during the
    initialization.
    """

    @wraps(init_func)
    def wrapper(*args, **kwargs):
        class_name = init_func.__qualname__.split(".")[0]
        logger.info("Initializing {}:".format(class_name))
        with indent_logs():
            return init_func(*args, **kwargs)

    return wrapper


# def indent_func_log(func):
#     """
#     Simple decorator to indent all logs that are created during the execution
#     of a function.
#     """
#
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         with indent_logs():
#             return func(*args, **kwargs)
#
#     return wrapper


# class Logger():
#     def __init__(self):
#         loguru.logger.remove()
#         duplicate_filter = DuplicateFilter()
#         loguru.logger.add(write_tqdm, colorize=True, format=formatter, filter=duplicate_filter)
#
#         return loguru.logger

logger = setup_logger()
