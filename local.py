#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the local computations for decentralized regression
(normal equation) including decentralized statistic calculation
"""
import warnings
warnings.simplefilter("ignore")

import logging
import ujson as json
import numpy as np
import os
import pandas as pd
import sys
import regression as reg
from parsers import vbm_parser
from local_ancillary import add_site_covariates
from local_ancillary import mean_and_len_y, local_stats_to_dict_numba
from memory_profiler import profile

fp = open('/output/memory_log', 'a+')

@profile(stream=fp)
def local_0(args):

    logging.basicConfig(filename='/output/app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    logging.debug('This is a debug message')
    logging.info('This is an info message')
    logging.warning('This is a warning message')
    logging.error('This is an error message')
    logging.critical('This is a critical message')

    computation_output_dict = {
        "output": {
            "computation_phase": "local_0"
        },
        "cache": {},
    }

    args_file = os.path.join(args['state']['cacheDirectory'], 'args_file')
    with open(args_file, 'w') as f:
        json.dump(args, f)

    return json.dumps(computation_output_dict)


@profile(stream=fp)
def local_1(args):

    args_file = os.path.join(args['state']['cacheDirectory'], 'args_file')

    with open(args_file, 'r') as f:
        original_args = json.load(f)

    lamb = original_args['input']['lambda']

    X, y = vbm_parser(original_args)
    y_labels = list(y.columns)
    
    meanY_vector, lenY_vector = mean_and_len_y(y)

    _, local_stats_list = local_stats_to_dict_numba(args, X, y)

    augmented_X = add_site_covariates(args, X)

    X_labels = list(augmented_X.columns)

    biased_X = augmented_X.values

    XtransposeX_local = np.matmul(np.matrix.transpose(biased_X), biased_X)
    Xtransposey_local = np.matmul(np.matrix.transpose(biased_X), y)
    
    output_dict = {
        "XtransposeX_local": XtransposeX_local.tolist(),
        "Xtransposey_local": Xtransposey_local.tolist(),
        "mean_y_local": meanY_vector,
        "count_local": lenY_vector,
        "local_stats_list": local_stats_list,
        "X_labels": X_labels,
        "y_labels": y_labels,
        "lambda": lamb
    }
    cache_dict = {
        "covariates": augmented_X.to_json(orient='split'),
    }

    local_output = os.path.join(args['state']['transferDirectory'],
                                'local_output')
    with open(local_output, 'w') as f:
        json.dump(output_dict, f)

    computation_output_dict = {
        "output": {
            "computation_phase": "local_1"
        },
        "cache": cache_dict,
    }

    return json.dumps(computation_output_dict)


@profile(stream=fp)
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

    args_file = os.path.join(args['state']['cacheDirectory'], 'args_file')

    with open(args_file, 'r') as f:
        original_args = json.load(f)

    X = pd.read_json(cache_list["covariates"], orient='split')
    (_, y) = vbm_parser(original_args)
    biased_X = np.array(X)

    avg_beta_vector = input_list["avg_beta_vector"]
    mean_y_global = input_list["mean_y_global"]

    y = y.values.astype('float64')

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

    local_output = os.path.join(args['state']['transferDirectory'],
                                'local_output')
    with open(local_output, 'w') as f:
        json.dump(output_dict, f)

    computation_output = {
        "output": {
            "computation_phase": 'local_2'
        },
        "cache": {}
    }

    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(reg.list_recursive(parsed_args, 'computation_phase'))

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
