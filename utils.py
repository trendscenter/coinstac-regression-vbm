#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 20:37:34 2019

@author: hgazula
"""


def list_recursive(parsed_dict, key):
    """Recursively searched to find the value of the key."""
    for k, val in parsed_dict.items():
        if isinstance(val, dict):
            for found in list_recursive(val, key):
                yield found
        if k == key:
            yield val
