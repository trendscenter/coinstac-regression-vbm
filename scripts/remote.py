#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the remote computations for decentralized
regression with decentralized statistic calculation
"""
import os
import shutil
import sys
import warnings

import numpy as np
import pandas as pd

import simplejson as json
import jsonpickle

from scripts.ancillary import (
    encode_png,
    loadBin,
    print_beta_images,
    print_pvals,
    print_r2_image,
    saveBin,
)
from scripts.nipype_utils import calculate_mask
from scripts.remote_ancillary import remote_stats, return_uniques_and_counts
from scripts.rw_utils import read_file
from scipy import stats
from scripts.utils import list_recursive, log

warnings.simplefilter("ignore")
OUTPUT_FROM_LOCAL = "local_output"


def remote_0(args):
    """The first function in the remote computation chain"""
    calculate_mask(args)
    input_ = args["input"]
    site_info = {site: input_[site]["categorical_dict"] for site in input_.keys()}
    log("args received to remote_0 : "+str(args), args["state"])

    df = pd.DataFrame.from_dict(site_info)
    covar_keys, unique_count = return_uniques_and_counts(df)

    computation_output_dict = {
        "output": {
            "covar_keys": jsonpickle.encode(covar_keys, unpicklable=False),
            "global_unique_count": unique_count,
            "mask": "mask.nii",
            "computation_phase": "remote_0",
        },
        "cache": {},
    }

    return computation_output_dict


def remote_1(args):
    """The second function in the local computation chain"""
    input_ = args["input"]
    state_ = args["state"]
    input_dir = state_["baseDirectory"]
    cache_dir = state_["cacheDirectory"]

    site_list = input_.keys()
    user_id = list(site_list)[0]

    input_list = dict()

    for site in site_list:
        file_name = os.path.join(input_dir, site, OUTPUT_FROM_LOCAL)
        input_list[site] = read_file(args, "input", file_name)

    X_labels = input_list[user_id]["X_labels"]

    all_local_stats_dicts = [
        input_list[site]["local_stats_list"] for site in input_list
    ]

    beta_vector_0 = sum(
        [
            loadBin(
                os.path.join(input_dir, site, input_list[site]["XtransposeX_local"])
            )
            for site in input_list
        ]
    )

    beta_vector_1 = sum(
        [
            loadBin(
                os.path.join(input_dir, site, input_list[site]["Xtransposey_local"])
            )
            for site in input_list
        ]
    )

    all_lambdas = [input_list[site]["lambda"] for site in input_list]

    if np.unique(all_lambdas).shape[0] != 1:
        raise Exception("Unequal lambdas at local sites")

    try:
        avg_beta_vector = np.transpose(
            np.dot(np.linalg.inv(beta_vector_0), beta_vector_1)
        )
    except np.linalg.LinAlgError:
        cond = np.linalg.cond(X.T @ X);
        raise Exception(f"X.^T*X matrix at remote is Singular with condition number: {cond}")

    mean_y_local = [input_list[site]["mean_y_local"] for site in input_list]
    count_y_local = [np.array(input_list[site]["count_local"]) for site in input_list]
    mean_y_global = np.array(mean_y_local) * np.array(count_y_local)
    mean_y_global = np.sum(mean_y_global, axis=0) / np.sum(count_y_local, axis=0)

    dof_global = sum(count_y_local) - avg_beta_vector.shape[1]

    saveBin(
        os.path.join(args["state"]["transferDirectory"], "avg_beta_vector.npy"),
        avg_beta_vector,
    )
    saveBin(
        os.path.join(args["state"]["transferDirectory"], "mean_y_global.npy"),
        mean_y_global,
    )

    saveBin(
        os.path.join(args["state"]["cacheDirectory"], "avg_beta_vector.npy"),
        avg_beta_vector,
    )

    output_dict = {
        "avg_beta_vector": "avg_beta_vector.npy",
        "mean_y_global": "mean_y_global.npy",
        "computation_phase": "remote_1",
    }

    cache_dict = {
        "avg_beta_vector": "avg_beta_vector.npy",
        "dof_global": dof_global.tolist(),
        "X_labels": X_labels,
        "local_stats_dict": all_local_stats_dicts,
    }

    computation_output_dict = {"output": output_dict, "cache": cache_dict}

    file_name = os.path.join(cache_dir, "remote_cache")
    with open(file_name, "w") as file_h:
        input_list[site] = json.dump(cache_dict, file_h)

    return computation_output_dict


def remote_2(args):
    """
    Computes the global model fit statistics, r_2_global, ts_global, ps_global

    Args:
        args (dictionary): {"input": {
                                "SSE_local": ,
                                "SST_local": ,
                                "varX_matrix_local": ,
                                "computation_phase":
                                },
                            "cache":{},
                            }

    Returns:
        computation_output (json) : {"output": {
                                        "avg_beta_vector": ,
                                        "beta_vector_local": ,
                                        "r_2_global": ,
                                        "ts_global": ,
                                        "ps_global": ,
                                        "dof_global":
                                        },
                                    "success":
                                    }
    Comments:
        Generate the local fit statistics
            r^2 : goodness of fit/coefficient of determination
                    Given as 1 - (SSE/SST)
                    where   SSE = Sum Squared of Errors
                            SST = Total Sum of Squares
            t   : t-statistic is the coefficient divided by its standard error.
                    Given as beta/std.err(beta)
            p   : two-tailed p-value (The p-value is the probability of
                  seeing a result as extreme as the one you are
                  getting (a t value as large as yours)
                  in a collection of random data in which
                  the variable had no effect.)

    """
    cache_ = args["cache"]
    state_ = args["state"]
    input_dir = state_["baseDirectory"]
    cache_dir = state_["cacheDirectory"]

    input_list = dict()
    site_list = args["input"].keys()
    for site in site_list:
        file_name = os.path.join(input_dir, site, OUTPUT_FROM_LOCAL)
        with open(file_name, "r") as file_h:
            input_list[site] = json.load(file_h)

    cache_list = read_file(args, "cache", "remote_cache")

    X_labels = args["cache"]["X_labels"]

    all_local_stats_dicts = cache_["local_stats_dict"]

    avg_beta_vector = loadBin(os.path.join(cache_dir, cache_list["avg_beta_vector"]))
    dof_global = cache_list["dof_global"]

    SSE_global = sum([np.array(input_list[site]["SSE_local"]) for site in input_list])
    SST_global = sum([np.array(input_list[site]["SST_local"]) for site in input_list])
    varX_matrix_global = sum(
        [np.array(input_list[site]["varX_matrix_local"]) for site in input_list]
    )

    r_squared_global = 1 - (SSE_global / SST_global)
    MSE = SSE_global / np.array(dof_global)
    ts_global = remote_stats(MSE, varX_matrix_global, avg_beta_vector)
    ps_global = [2 * stats.t.sf(np.abs(t), df) for t, df in zip(ts_global, dof_global)]

    print_pvals(args, ps_global, ts_global, X_labels)
    print_beta_images(args, avg_beta_vector, X_labels)
    print_r2_image(args, r_squared_global)

    # Block of code to print local stats as well
    sites = [site for site in input_list]

    all_local_stats_dicts = dict(zip(sites, all_local_stats_dicts))

    # List all output files (global stats)
    global_files = sorted(os.listdir(state_["outputDirectory"]))

    # move global stats to transfer directory to be moved to local sites
    os.mkdir(os.path.join(state_["transferDirectory"], "global_stats"))
    for f in global_files:
        shutil.move(
            os.path.join(state_["outputDirectory"], f),
            os.path.join(state_["transferDirectory"], "global_stats", f),
        )

    # Get list of global pngs files
    global_png_files = [
        f"global_stats/{file}" for file in global_files if file.endswith(".png")
    ]

    # Get list of local png files
    local_png_files = {
        site: [file for file in site_files if file.endswith(".png")]
        for site, site_files in all_local_stats_dicts.items()
    }

    # Gather local and global png files (for display in UI)
    output_dict = {"local_stats": local_png_files, "global_stats": global_png_files}

    computation_output_dict = {"output": output_dict, "success": True}

    return computation_output_dict


def start(PARAM_DICT):
    PHASE_KEY = list(list_recursive(PARAM_DICT, "computation_phase"))

    if "local_0" in PHASE_KEY:
        return remote_0(PARAM_DICT)
    elif "local_1" in PHASE_KEY:
        return remote_1(PARAM_DICT)
    elif "local_2" in PHASE_KEY:
        return remote_2(PARAM_DICT)
    else:
        raise ValueError("Error occurred at Remote")
