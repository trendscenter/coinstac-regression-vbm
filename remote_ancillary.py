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


def extract_sites(site_dict):
    """Extracts and returns sub-site identifiers from each site
    """
    for _, site in site_dict.items():
        if isinstance(site, dict):
            for found in extract_sites(site):
                yield found
        else:
            yield site
