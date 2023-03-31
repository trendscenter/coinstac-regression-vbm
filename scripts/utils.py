#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 20:37:34 2019

@author: hgazula
"""
import logging
import os

def list_recursive(parsed_dict, key):
    """Recursively searched to find the value of the key."""
    for k, val in parsed_dict.items():
        if isinstance(val, dict):
            for found in list_recursive(val, key):
                yield found
        if k == key:
            yield val


def log(msg, state):
    # create logger with 'spam_application'
    logger = logging.getLogger(state["clientId"])
    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    if len(logger.handlers) == 0:
        filename = os.path.join(state["outputDirectory"], 'COINSTAC_%s.log' % state["clientId"])
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.INFO)
        # create console handler with a higher log level
        logger.addHandler(fh)
    logger.info(msg)
