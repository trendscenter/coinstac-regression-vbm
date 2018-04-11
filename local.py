#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the local computations for decentralized regression with
normal equation including decentralized statistic calculation
"""
import json
import numpy as np
import pandas as pd
import sys
import regression as reg
import warnings
from parsers import vbm_parser

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statsmodels.api as sm


def local_0(args):
    input_list = args["input"]
    lamb = input_list["lambda"]

    (X, y) = vbm_parser(args)
    y = y.loc[:, 0:1000]  # comment this line to run on all voxels
    y_labels = ['{}_{}'.format('voxel', str(i)) for i in y.columns]

    computation_output_dict = {
        "output": {
            "computation_phase": "local_0"
        },
        "cache": {
            "covariates": X.values.tolist(),
            "dependents": y.values.tolist(),
            "lambda": lamb,
            "y_labels": y_labels
        },
    }

    return json.dumps(computation_output_dict)


def local_1(args):
    lamb = args["cache"]["lambda"]
    X = args["cache"]["covariates"]
    y = args["cache"]["dependents"]
    y_labels = args["cache"]["y_labels"]
    y = pd.DataFrame(y, columns=y_labels)

    biased_X = sm.add_constant(X)
    meanY_vector, lenY_vector = [], []

    local_params = []
    local_sse = []
    local_pvalues = []
    local_tvalues = []
    local_rsquared = []

    for column in y.columns:
        curr_y = y[column].values
        meanY_vector.append(np.mean(curr_y))
        lenY_vector.append(len(y))

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

    # +++++++++++++++++++++ Adding site covariate columns +++++++++++++++++++++
    site_covar_list = args["input"]["site_covar_list"]

    site_matrix = np.zeros(
        (np.array(X).shape[0], len(site_covar_list)), dtype=int)
    site_df = pd.DataFrame(site_matrix, columns=site_covar_list)

    select_cols = [
        col for col in site_df.columns if args["state"]["clientId"] in col
    ]

    site_df[select_cols] = 1
    biased_X = np.concatenate((biased_X, site_df.values), axis=1)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    XtransposeX_local = np.matmul(np.matrix.transpose(biased_X), biased_X)
    Xtransposey_local = np.matmul(np.matrix.transpose(biased_X), y)

    computation_output_dict = {
        "output": {
            "XtransposeX_local": XtransposeX_local.tolist(),
            "Xtransposey_local": Xtransposey_local.tolist(),
            "mean_y_local": meanY_vector,
            "count_local": lenY_vector,
            "local_stats_list": local_stats_list,
            "y_labels": y_labels,
            "computation_phase": "local_1"
        },
        "cache": {
            "covariates": biased_X.tolist(),
            "dependents": y.values.tolist(),
            "lambda": lamb
        },
    }
    return json.dumps(computation_output_dict)


def local_2(args):
    """Computes the SSE_local, SST_local and varX_matrix_local
    Args:
        args (dictionary): {"input": {
                                "avg_beta_vector": ,
                                "mean_y_global": ,
                                "computation_phase":
                                },
                            "cache": {
                                "covariates": ,
                                "dependents": ,
                                "lambda": ,
                                "dof_local": ,
                                }
                            }
    Returns:
        computation_output (json): {"output": {
                                        "SSE_local": ,
                                        "SST_local": ,
                                        "varX_matrix_local": ,
                                        "computation_phase":
                                        }
                                    }
    Comments:
        After receiving  the mean_y_global, calculate the SSE_local,
        SST_local and varX_matrix_local
    """
    cache_list = args["cache"]
    input_list = args["input"]

    X = cache_list["covariates"]
    y = cache_list["dependents"]
    biased_X = sm.add_constant(X)

    avg_beta_vector = input_list["avg_beta_vector"]
    mean_y_global = input_list["mean_y_global"]

    y = pd.DataFrame(y)
    SSE_local, SST_local = [], []
    for index, column in enumerate(y.columns):
        curr_y = y[column].values
        SSE_local.append(
            reg.sum_squared_error(biased_X, curr_y, avg_beta_vector))
        SST_local.append(
            np.sum(np.square(np.subtract(curr_y, mean_y_global[index]))))

    varX_matrix_local = np.dot(biased_X.T, biased_X)

    computation_output = {
        "output": {
            "SSE_local": SSE_local,
            "SST_local": SST_local,
            "varX_matrix_local": varX_matrix_local.tolist(),
            "computation_phase": 'local_2'
        },
        "cache": {}
    }

    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(reg.listRecursive(parsed_args, 'computation_phase'))

    if not phase_key:
        computation_output = local_0(parsed_args)
        sys.stdout.write(computation_output)
    elif "remote_0" in phase_key:
        computation_output = local_1(parsed_args)
        sys.stdout.write(computation_output)
    elif "remote_1" in phase_key:
        computation_output = local_2(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Local")
