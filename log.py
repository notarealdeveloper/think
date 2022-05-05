#!/usr/bin/env python3

__all__ = [
    'set_log_level',
]

import os
import sys
import logging
import threading

def set_log_level(level, name='think'):
    logger = logging.getLogger(name)
    levelnum = getattr(logging, level.upper())
    logger.setLevel(levelnum)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            continue
        handler.setLevel(levelnum)
    return logger


def get_logger(name):

    # create logger
    logger = logging.getLogger(name)

    # set level
    level = logging.WARNING
    logger.setLevel(level)

    # create formatter
    format = '%(asctime)s.%(msecs)03d:%(name)s:%(funcName)s:%(levelname)s:%(message)s'
    formatter = logging.Formatter(format, datefmt='%Y-%m-%d_%H:%M:%S')

    # create stream handler
    stream = threading.main_thread()._stderr
    #stream = sys.stdout
    sh = logging.StreamHandler(stream)
    sh.setLevel(level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # create file handler and set level to debug
    if file := os.getenv("THINK"):
        fh = logging.FileHandler(file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


root = get_logger('think')
