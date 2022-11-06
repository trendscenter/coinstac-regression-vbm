#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 20:00:26 2019

@author: hgazula
"""
import os

import simplejson as json


def return_file(spec, location, file_name):
    cache_dir = spec["state"]["cacheDirectory"]
    input_dir = spec["state"]["baseDirectory"]
    output_dir = spec["state"]["transferDirectory"]

    loc_dict = {"cache": cache_dir, "input": input_dir, "output": output_dir}

    args_file = os.path.join(loc_dict[location], file_name)

    return args_file


def write_file(spec, contents=None, location=None, file_name=None):
    """Write contents to a file"""
    args_file = return_file(spec, location, file_name)

    if contents is None:
        contents = spec

    with open(args_file, "w") as file_h:
        json.dump(contents, file_h)


def read_file(spec, location, file_name):
    """Read contents from a file"""
    args_file = return_file(spec, location, file_name)

    with open(args_file, "r") as file_h:
        output = json.load(file_h)

    return output
