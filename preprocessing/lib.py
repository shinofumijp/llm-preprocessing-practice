from doctest import debug
from typing import Generator
import os
from logging import getLogger, WARN, DEBUG,  ERROR, INFO, StreamHandler, FileHandler, Filter, Formatter, warn

from numpy import info


def readlines(file: str) -> Generator[str, None, None]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            yield line


class Logger:
    class InfoFilter(Filter):
        def filter(self, record):
            return record.levelno == INFO

    class WarnFilter(Filter):
        def filter(self, record):
            return record.levelno == WARN

    @staticmethod
    def get_logger(name: str, logdir: str, verbose: bool = False) -> None:
        logger = getLogger(name)
        logger.setLevel(DEBUG)
        formatter = Formatter('[%(levelname)s]\t%(asctime)s\t%(name)s\t%(message)s')

        if verbose:
            debug_handler = StreamHandler()
            debug_handler.setLevel(DEBUG)
            debug_handler.setFormatter(formatter)
            logger.addHandler(debug_handler)

        err_handler = FileHandler(os.path.join(logdir, "error.log"))
        err_handler.setLevel(ERROR)
        err_handler.setFormatter(formatter)
        logger.addHandler(err_handler)

        warn_handler = FileHandler(os.path.join(logdir, "warn.log"))
        warn_handler.setLevel(WARN)
        warn_handler.addFilter(Logger.WarnFilter())
        warn_handler.setFormatter(formatter)
        logger.addHandler(warn_handler)

        info_handler = FileHandler(os.path.join(logdir, "info.log"))
        info_handler.setLevel(INFO)
        info_handler.addFilter(Logger.InfoFilter())
        info_handler.setFormatter(formatter)
        logger.addHandler(info_handler)

        return logger
