#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 22:28:11 2018

@author: Harshvardhan
"""
import os
import warnings

import numpy as np
import pandas as pd
import scipy as sp
from numba import jit, prange

from ancillary import encode_png, print_beta_images, print_pvals
from parsers import parse_for_covar_info, perform_encoding

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statsmodels.api as sm

MASK = os.path.join('/computation', 'mask_2mm.nii')


def mean_and_len_y(y):
    """Caculate the mean and length of each y vector"""
    meanY_vector = y.mean(axis=0)
    #    lenY_vector = y.count(axis=0)
    lenY_vector = np.count_nonzero(~np.isnan(y), axis=0)

    return meanY_vector, lenY_vector


@jit(nopython=True)
def gather_local_stats(X, y):
    """Calculate local statistics"""
    size_y = y.shape[1]

    params = np.zeros((X.shape[1], size_y))
    sse = np.zeros(size_y)
    tvalues = np.zeros((X.shape[1], size_y))
    rsquared = np.zeros(size_y)

    for voxel in prange(size_y):
        curr_y = y[:, voxel]
        beta_vector = np.linalg.inv(X.T @ X) @ (X.T @ curr_y)
        params[:, voxel] = beta_vector

        curr_y_estimate = np.dot(beta_vector, X.T)

        SSE_global = np.linalg.norm(curr_y - curr_y_estimate)**2
        SST_global = np.sum(np.square(curr_y - np.mean(curr_y)))

        sse[voxel] = SSE_global
        r_squared_global = 1 - (SSE_global / SST_global)
        rsquared[voxel] = r_squared_global

        dof_global = len(curr_y) - len(beta_vector)

        MSE = SSE_global / dof_global
        var_covar_beta_global = MSE * np.linalg.inv(X.T @ X)
        se_beta_global = np.sqrt(np.diag(var_covar_beta_global))
        ts_global = beta_vector / se_beta_global

        tvalues[:, voxel] = ts_global

    return (params, sse, tvalues, rsquared, dof_global)


def local_stats_to_dict_numba(args, X, y):
    """Wrap local statistics into a dictionary to be sent to the remote"""
    X1 = sm.add_constant(X)

    X_labels = list(X1.columns)

    X1 = X1.values.astype('float64')
    y1 = y.astype('float64')

    params, _, tvalues, _, dof_global = gather_local_stats(X1, y1)

    pvalues = 2 * sp.stats.t.sf(np.abs(tvalues), dof_global)

    #    keys = ["beta", "sse", "pval", "tval", "rsquared"]
    #
    #    values1 = pd.DataFrame(
    #        list(
    #            zip(params.T.tolist(), sse.tolist(), pvalues.T.tolist(),
    #                tvalues.T.tolist(), rsquared.tolist())),
    #        columns=keys)
    #
    #    local_stats_list = values1.to_dict(orient='records')

    beta_vector = params.T.tolist()

    print_pvals(args, pvalues.T, tvalues.T, X_labels)
    print_beta_images(args, beta_vector, X_labels)

    local_stats_list = encode_png(args)

    return beta_vector, local_stats_list


def local_stats_to_dict(X, y):
    """Calculate local statistics"""
    y_labels = list(y.columns)

    biased_X = sm.add_constant(X)

    local_params = []
    local_sse = []
    local_pvalues = []
    local_tvalues = []
    local_rsquared = []

    for column in y.columns:
        curr_y = list(y[column])

        # Printing local stats as well
        model = sm.OLS(curr_y, biased_X.astype(float)).fit()
        local_params.append(model.params)
        local_sse.append(model.ssr)
        local_pvalues.append(model.pvalues)
        local_tvalues.append(model.tvalues)
        local_rsquared.append(model.rsquared_adj)

    keys = ["beta", "sse", "pval", "tval", "rsquared"]
    local_stats_list = []

    for index, _ in enumerate(y_labels):
        values = [
            local_params[index].tolist(), local_sse[index],
            local_pvalues[index].tolist(), local_tvalues[index].tolist(),
            local_rsquared[index]
        ]
        local_stats_dict = {key: value for key, value in zip(keys, values)}
        local_stats_list.append(local_stats_dict)

        beta_vector = [l.tolist() for l in local_params]

    return beta_vector, local_stats_list


def merging_globals(X, site_covar_dict, dict_, key):
    """Merge the actual data frame with the created dummy matrix
    """
    site_covar_dict.rename(index=dict(enumerate(dict_[key])), inplace=True)
    site_covar_dict.index.name = key
    site_covar_dict.reset_index(level=0, inplace=True)
    X = X.merge(site_covar_dict, left_on=key, right_on=key)
    X.drop(columns=key, inplace=True)

    return X


# TODO: Right now this only works for 'site' covariate. Need to extend to other
#    categorial covariates as well
def add_site_covariates(args, original_args, X):
    """Add site covariates based on information gathered from all sites
    """
    input_ = args["input"]
    all_sites = input_["site_list"]
    glob_uniq_ct = input_["global_unique_count"]

    # Read original covariate_info
    X, _ = parse_for_covar_info(original_args)

    to_exclude = []
    for key, val in glob_uniq_ct.items():
        if val == 1:
            X.drop(columns=key, inplace=True)
        elif val == 2:
            covar_dict = pd.get_dummies(all_sites[key],
                                        prefix=key,
                                        drop_first=True)
            X = merging_globals(X, covar_dict, all_sites, key)
            to_exclude.append(key)

        else:
            covar_dict = pd.get_dummies(all_sites[key],
                                        prefix=key,
                                        drop_first=False)
            X = merging_globals(X, covar_dict, all_sites, key)
            to_exclude.append(key)

    biased_X = perform_encoding(X, exclude_cols=tuple(to_exclude))

    return biased_X
