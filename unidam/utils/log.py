#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 11:39:18 2017

@author: mints
"""
import logging
import colorlog


def get_logger(name, screen_output=True, log_filename=''):
    """
    Create and return a logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if len(logger.handlers) == 0:
        if screen_output:
            handler = logging.StreamHandler()
            handler.setFormatter(colorlog.ColoredFormatter(
                '%(asctime)-6s: %(name)s - %(log_color)s%(levelname)s%(reset)s - %(message)s'))
            logger.addHandler(handler)
        if log_filename != '':
            handler = logging.FileHandler(log_filename)
            handler.setFormatter(logging.Formatter(
                '%(asctime)-6s: %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
    return logger
