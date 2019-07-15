#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the local computations for decentralized regression
(normal equation) including decentralized statistic calculation
"""
import sys
import warnings

import numpy as np
import pandas as pd
import ujson as json
from numba import jit

import regression as reg
from local_ancillary import (add_site_covariates, local_stats_to_dict_numba,
                             mean_and_len_y)
from parsers import parse_for_site, vbm_parser, parse_for_covar_info
from rw_utils import read_file, write_file

warnings.simplefilter("ignore")


@jit(nopython=True)
def calc_XtransposeX_local(biased_X):
    """Calculates X.T * X
    """
    return biased_X.T @ biased_X


def calc_Xtransposey_local(biased_X, y):
    """Calculates X.T * y
    """
    biased_X = biased_X.astype('float64')
    y = y.astype('float64')

    @jit(nopython=True)
    def mult(a, b):
        """Multiplies two matrices"""
        return a @ b

    return mult(biased_X.T, y)


def local_0(args):
    """ The first function in the local computation chain
    """
    site_dict = parse_for_site(args)

    output_dict = {"site_dict": site_dict, "computation_phase": "local_0"}
    cache_dict = {}

    computation_output_dict = {
        "output": output_dict,
        "cache": cache_dict,
    }

    write_file(args, args, 'cache', 'args_file')

    return json.dumps(computation_output_dict)


def local_1(args):
    """ The second function in the local computation chain
    """
    original_args = read_file(args, 'cache', 'args_file')
    regularizer_l2 = original_args['input']['lambda']

    # Local Statistics
    X, y = vbm_parser(original_args, "local")
    meanY_vector, lenY_vector = mean_and_len_y(y)
    _, local_stats_list = local_stats_to_dict_numba(args, X, y)

    augmented_X = add_site_covariates(args, original_args, X)
    X_labels = list(augmented_X.columns)
    biased_X = augmented_X.values

    #    XtransposeX_local = np.matmul(np.matrix.transpose(biased_X), biased_X)
    #    Xtransposey_local = np.matmul(np.matrix.transpose(biased_X), y)

    XtransposeX_local = calc_XtransposeX_local(biased_X)
    Xtransposey_local = calc_Xtransposey_local(biased_X, y)

    output_dict = {
        "XtransposeX_local": XtransposeX_local.tolist(),
        "Xtransposey_local": Xtransposey_local.tolist(),
        "mean_y_local": meanY_vector.tolist(),
        "count_local": lenY_vector.tolist(),
        "local_stats_list": local_stats_list,
        "X_labels": X_labels,
        "lambda": regularizer_l2
    }
    cache_dict = {
        "covariates": augmented_X.to_json(orient='split'),
    }

    #    local_output = os.path.join(args['state']['transferDirectory'],
    #                                'local_output')
    #    with open(local_output, 'w') as file_h:
    #        json.dump(output_dict, file_h)

    write_file(args, output_dict, 'output', 'local_output')

    computation_output_dict = {
        "output": {
            "computation_phase": "local_1"
        },
        "cache": cache_dict,
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

    #    args_file = os.path.join(args['state']['cacheDirectory'], 'args_file')
    #
    #    with open(args_file, 'r') as file_h:
    #        original_args = json.load(file_h)
    #
    original_args = read_file(args, 'cache', 'args_file')

    X = pd.read_json(cache_list["covariates"], orient='split')
    (_, y) = vbm_parser(original_args)
    biased_X = np.array(X)

    avg_beta_vector = input_list["avg_beta_vector"]
    mean_y_global = input_list["mean_y_global"]

    y = y.astype('float64')

    SSE_local, SST_local = [], []
    for voxel in range(y.shape[1]):
        curr_y = y[:, voxel]
        SSE_local.append(
            reg.sum_squared_error(biased_X, curr_y, avg_beta_vector[voxel]))
        SST_local.append(
            np.sum(np.square(np.subtract(curr_y, mean_y_global[voxel]))))

    varX_matrix_local = np.dot(biased_X.T, biased_X)

    output_dict = {
        "SSE_local": SSE_local,
        "SST_local": SST_local,
        "varX_matrix_local": varX_matrix_local.tolist()
    }

    write_file(args, output_dict, 'output', 'local_output')

    output_dict = {"computation_phase": 'local_2'}
    cache_dict = {}
    computation_output_dict = {"output": output_dict, "cache": cache_dict}

    return json.dumps(computation_output_dict)


if __name__ == '__main__':

    PARSED_ARGS = json.loads(sys.stdin.read())
    PHASE_KEY = list(reg.list_recursive(PARSED_ARGS, 'computation_phase'))

    if not PHASE_KEY:
        sys.stdout.write(local_0(PARSED_ARGS))
    elif "remote_0" in PHASE_KEY:
        sys.stdout.write(local_1(PARSED_ARGS))
    elif "remote_1" in PHASE_KEY:
        sys.stdout.write(local_2(PARSED_ARGS))
    else:
        raise ValueError("Error occurred at Local")
