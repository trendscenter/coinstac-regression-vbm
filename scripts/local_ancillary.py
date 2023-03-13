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
import simplejson as json
from numba import jit, prange

import scipy as sp
from scripts.ancillary import encode_png, print_beta_images, print_pvals, print_r2_image
from scripts.nipype_utils import nifti_to_data
from scripts.parsers import perform_encoding

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statsmodels.api as sm


def mean_and_len_y(y):
    """Caculate the mean and length of each y vector"""
    meanY_vector = y.mean(axis=0)
    #    lenY_vector = y.count(axis=0)
    lenY_vector = np.count_nonzero(~np.isnan(y), axis=0)

    return meanY_vector, lenY_vector


@jit(nopython=True)
def gather_local_stats_helper(X, y, pinv):
    """Calculate local statistics"""
    size_y = y.shape[1]

    params = np.zeros((X.shape[1], size_y))
    sse = np.zeros(size_y)
    tvalues = np.zeros((X.shape[1], size_y))
    rsquared = np.zeros(size_y)

    for voxel in prange(size_y):
        curr_y = y[:, voxel]
        beta_vector = pinv @ (X.T @ curr_y)
        params[:, voxel] = beta_vector

        curr_y_estimate = np.dot(beta_vector, X.T)

        SSE_global = np.linalg.norm(curr_y - curr_y_estimate) ** 2
        SST_global = np.sum(np.square(curr_y - np.mean(curr_y)))

        sse[voxel] = SSE_global
        r_squared_global = 1 - (SSE_global / SST_global)
        rsquared[voxel] = r_squared_global

        dof_global = len(curr_y) - len(beta_vector)

        MSE = SSE_global / dof_global
        var_covar_beta_global = MSE * pinv
        se_beta_global = np.sqrt(np.diag(var_covar_beta_global))
        ts_global = beta_vector / se_beta_global

        tvalues[:, voxel] = ts_global

    return (params, sse, tvalues, rsquared, dof_global)


def gather_local_stats(X, y):
    """Calculate local statistics"""
    try:
        pinv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        cond = np.linalg.cond(X.T @ X);
        raise Exception(f"X.^T*X matrix at local is Singular with condition number: {cond}")

    return gather_local_stats_helper(X, y, pinv)


def local_stats_to_dict_numba(args, X, y):
    """Wrap local statistics into a dictionary to be sent to the remote"""
    X_labels = list(X.columns)

    X1 = X.values.astype("float64")

    params, _, tvalues, rsquared, dof_global = gather_local_stats(X1, y)

    pvalues = 2 * sp.stats.t.sf(np.abs(tvalues), dof_global)

    beta_vector = params.T.tolist()

    print_pvals(args, pvalues.T, tvalues.T, X_labels)
    print_beta_images(args, beta_vector, X_labels)
    print_r2_image(args, rsquared)

    local_stats_list = sorted(os.listdir(args["state"]["outputDirectory"]))

    return beta_vector, local_stats_list


def local_stats_to_dict(X, y):
    """Calculate local statistics"""
    y_labels = list(y.columns)

    biased_X = X

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
            local_params[index].tolist(),
            local_sse[index],
            local_pvalues[index].tolist(),
            local_tvalues[index].tolist(),
            local_rsquared[index],
        ]
        local_stats_dict = {key: value for key, value in zip(keys, values)}
        local_stats_list.append(local_stats_dict)

        beta_vector = [l.tolist() for l in local_params]

    return beta_vector, local_stats_list


def merging_globals(args, X, site_covar_dict, dict_, key):
    """Merge the actual data frame with the created dummy matrix"""
    site_covar_dict.rename(index=dict(enumerate(dict_[key])), inplace=True)
    site_covar_dict.index.name = key
    site_covar_dict.reset_index(level=0, inplace=True)
    X = X.merge(site_covar_dict, on=key, how="left")
    X = X.drop(columns=key)

    return X


def add_site_covariates(args, X):
    """Add site covariates based on information gathered from all sites"""
    input_ = args["input"]
    all_sites = input_["covar_keys"]
    glob_uniq_ct = input_["global_unique_count"]

    all_sites = json.loads(all_sites)

    for key, val in glob_uniq_ct.items():
        if val == 1:
            X.drop(columns=key, inplace=True)
        else:
            covar_dict = pd.get_dummies(all_sites[key], prefix=key, drop_first=False)
            X = merging_globals(args, X, covar_dict, all_sites, key)

    X = X.dropna(axis=0, how="any")
    biased_X = sm.add_constant(X, has_constant="add")

    return biased_X


@jit(nopython=True)
def multiply(array_a, array_b):
    """Multiplies two matrices"""
    return array_a.T @ array_b


@jit(nopython=True)
def stats_calculation(X, y, avg_beta_vec, mean_y_global):
    """Calculate SSE and SST."""
    size_y = y.shape[1]
    sse_local = np.zeros(size_y)
    sst_local = np.zeros(size_y)

    for voxel in range(y.shape[1]):
        y1 = y[:, voxel]
        beta = avg_beta_vec[voxel]
        mean_y = mean_y_global[voxel]

        y1_estimate = np.dot(beta, X.T)
        sse_local[voxel] = np.linalg.norm(y1 - y1_estimate) ** 2
        sst_local[voxel] = np.sum(np.square(y1 - mean_y))

    return sse_local, sst_local


def vbm_parser(args, X):
    """Parse the nifti (.nii) specific inputspec.json and return the
    covariate matrix (X) as well the dependent matrix (y) as dataframes
    """
    y_info = nifti_to_data(args, X)
    encoded_covar_info = perform_encoding(args, X)

    return (encoded_covar_info, y_info)
