#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 14:56:41 2018

@author: Harshvardhan
"""
import warnings

import pandas as pd

warnings.simplefilter("ignore")


def get_stats_to_dict(col_names, *b):
    """Convert dataframe stats to dictionary
    """
    stats_df = pd.DataFrame(list(zip(*b)), columns=col_names)
    dict_list = stats_df.to_dict(orient='records')

    return dict_list


def return_uniques_and_counts(df):
    """Return unique-values of the categorical variables and their counts
    """
    keys, count = dict(), dict()
    for index, row in df.iterrows():
        flat_list = [item for sublist in row for item in sublist]
        keys[index] = set(flat_list)
        count[index] = len(set(flat_list))

    return keys, count
