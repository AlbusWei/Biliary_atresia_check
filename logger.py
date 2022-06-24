import datetime
import logging
import os
import sys

import paddle.distributed as dist

_logger = None


def init_logger(name='ppcls', log_file=None, log_level=logging.INFO):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified a FileHandler will also be added.
    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
    Returns:
        logging.Logger: The expected logger.
    """
    global _logger

    #  solve mutiple init issue when using paddleclas.py and engin.engin
    init_flag = False
    if _logger is None:
        _logger = logging.getLogger(name)
        init_flag = True

    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt="%Y/%m/%d %H:%M:%S")

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler._name = 'stream_handler'

    # add stream_handler when _logger dose not contain stream_handler
    for i, h in enumerate(_logger.handlers):
        if h.get_name() == stream_handler.get_name():
            break
        if i == len(_logger.handlers) - 1:
            _logger.addHandler(stream_handler)
    if init_flag:
        _logger.addHandler(stream_handler)

    if log_file is not None and dist.get_rank() == 0:
        log_file_folder = os.path.split(log_file)[0]
        os.makedirs(log_file_folder, exist_ok=True)
        file_handler = logging.FileHandler(log_file, 'a')
        file_handler.setFormatter(formatter)
        file_handler._name = 'file_handler'

        # add file_handler when _logger dose not contain same file_handler
        for i, h in enumerate(_logger.handlers):
            if h.get_name() == file_handler.get_name() and \
                    h.baseFilename == file_handler.baseFilename:
                break
            if i == len(_logger.handlers) - 1:
                _logger.addHandler(file_handler)

    if dist.get_rank() == 0:
        _logger.setLevel(log_level)
    else:
        _logger.setLevel(logging.ERROR)
    _logger.propagate = False


def log_at_trainer0(log):
    """
    logs will print multi-times when calling Fleet API.
    Only display single log and ignore the others.
    """

    def wrapper(fmt, *args):
        if dist.get_rank() == 0:
            log(fmt, *args)

    return wrapper


@log_at_trainer0
def info(fmt, *args):
    _logger.info(fmt, *args)


@log_at_trainer0
def debug(fmt, *args):
    _logger.debug(fmt, *args)


@log_at_trainer0
def warning(fmt, *args):
    _logger.warning(fmt, *args)


@log_at_trainer0
def error(fmt, *args):
    _logger.error(fmt, *args)


def scaler(name, value, step, writer):
    """
    This function will draw a scalar curve generated by the visualdl.
    Usage: Install visualdl: pip3 install visualdl==2.0.0b4
           and then:
           visualdl --logdir ./scalar --host 0.0.0.0 --port 8830
           to preview loss corve in real time.
    """
    if writer is None:
        return
    writer.add_scalar(tag=name, step=step, value=value)


def advertise():
    """
    Show the advertising message like the following:
    ===========================================================
    ==        PaddleClas is powered by PaddlePaddle !        ==
    ===========================================================
    ==                                                       ==
    ==   For more info please go to the following website.   ==
    ==                                                       ==
    ==       https://github.com/PaddlePaddle/PaddleClas      ==
    ===========================================================
    """
    copyright = "PaddleClas is powered by PaddlePaddle !"
    ad = "For more info please go to the following website."
    website = "https://github.com/PaddlePaddle/PaddleClas"
    AD_LEN = 6 + len(max([copyright, ad, website], key=len))

    info("\n{0}\n{1}\n{2}\n{3}\n{4}\n{5}\n{6}\n{7}\n".format(
        "=" * (AD_LEN + 4),
        "=={}==".format(copyright.center(AD_LEN)),
        "=" * (AD_LEN + 4),
        "=={}==".format(' ' * AD_LEN),
        "=={}==".format(ad.center(AD_LEN)),
        "=={}==".format(' ' * AD_LEN),
        "=={}==".format(website.center(AD_LEN)),
        "=" * (AD_LEN + 4), ))