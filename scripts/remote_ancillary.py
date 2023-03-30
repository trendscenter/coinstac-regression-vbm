#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 14:56:41 2018

@author: Harshvardhan
"""
import warnings

import numpy as np
import pandas as pd
from numba import jit, prange

warnings.simplefilter("ignore")


def get_stats_to_dict(col_names, *b):
    """Convert dataframe stats to dictionary"""
    stats_df = pd.DataFrame(list(zip(*b)), columns=col_names)
    dict_list = stats_df.to_dict(orient="records")

    return dict_list


def return_uniques_and_counts(df):
    """Return unique-values of the categorical variables and their counts"""
    keys, count = dict(), dict()
    keys = (
        df.iloc[:, :].sum(axis=1).apply(set).apply(sorted).to_dict()
    )  # adding all columns
    count = {k: len(v) for k, v in keys.items()}

    return keys, count


@jit(nopython=True)
def remote_stats(MSE, varX_matrix_global, avg_beta_vector):
    my_shape = avg_beta_vector.shape
    ts = np.zeros(my_shape)

    for voxel in prange(my_shape[0]):
        var_covar_beta_global = MSE[voxel] * np.linalg.inv(varX_matrix_global)
        se_beta_global = np.sqrt(np.diag(var_covar_beta_global))
        ts[voxel, :] = avg_beta_vector[voxel, :] / se_beta_global

    return ts
