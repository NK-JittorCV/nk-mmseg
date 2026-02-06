"""
Logger utilities for DFormer Jittor implementation
Adapted from PyTorch version for Jittor framework
"""

import os
import sys
import logging

from utils import jt_utils

_default_level_name = os.getenv("ENGINE_LOGGING_LEVEL", "INFO")
_default_level = logging.getLevelName(_default_level_name.upper())


class LogFormatter(logging.Formatter):
    """Custom log formatter with colors."""
    
    log_fout = None
    date_full = "[%(asctime)s %(lineno)d@%(filename)s:%(name)s] "
    date = "%(asctime)s "
    msg = "%(message)s"

    def format(self, record):
        """Format log record with colors."""
        if record.levelno == logging.DEBUG:
            mcl, mtxt = self._color_dbg, "DBG"
        elif record.levelno == logging.WARNING:
            mcl, mtxt = self._color_warn, "WRN"
        elif record.levelno == logging.ERROR:
            mcl, mtxt = self._color_err, "ERR"
        else:
            mcl, mtxt = self._color_normal, ""

        if mtxt:
            mtxt += " "

        if self.log_fout:
            self.__set_fmt(self.date_full + mtxt + self.msg)
            formatted = super(LogFormatter, self).format(record)
            return formatted

        self.__set_fmt(self._color_date(self.date) + mcl(mtxt + self.msg))
        formatted = super(LogFormatter, self).format(record)

        return formatted

    def __set_fmt(self, fmt):
        """Set format string."""
        if sys.version_info.major < 3:
            self._fmt = fmt
        else:
            self._style._fmt = fmt

    @staticmethod
    def _color_dbg(msg):
        return "\x1b[36m{}\x1b[0m".format(msg)

    @staticmethod
    def _color_warn(msg):
        return "\x1b[1;31m{}\x1b[0m".format(msg)

    @staticmethod
    def _color_err(msg):
        return "\x1b[1;4;31m{}\x1b[0m".format(msg)

    @staticmethod
    def _color_omitted(msg):
        return "\x1b[35m{}\x1b[0m".format(msg)

    @staticmethod
    def _color_normal(msg):
        return msg

    @staticmethod
    def _color_date(msg):
        return "\x1b[32m{}\x1b[0m".format(msg)


def get_logger(log_dir=None, log_file=None, formatter=None, level=_default_level, rank=0):
    """Get logger instance."""
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False

    # Console handler
    if rank == 0:  # Only main process logs to console
        plain_formatter = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%m/%d %H:%M:%S"
        )
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(plain_formatter)  # Always use plain formatter for now
        logger.addHandler(ch)

    # File handler
    if log_dir and log_file and rank == 0:
        jt_utils.ensure_dir(log_dir)
        log_path = os.path.join(log_dir, log_file)
        
        fh = logging.FileHandler(log_path)
        fh.setLevel(level)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
    """Setup logger for training."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        jt_utils.ensure_dir(save_dir)
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
