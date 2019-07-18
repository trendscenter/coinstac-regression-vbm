#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the remote computations for decentralized
regression with decentralized statistic calculation
"""
import os
import sys
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
import scipy as sp
import ujson as json
from nilearn.image import resample_to_img

import regression as reg
from ancillary import encode_png, print_beta_images, print_pvals
from rw_utils import read_file

warnings.simplefilter("ignore")

OUTPUT_FROM_LOCAL = 'local_output'


def return_uniques_and_counts(df):
    """Return unique-values of the categorical variables and their counts
    """
    keys, count = dict(), dict()
    for index, row in df.iterrows():
        flat_list = [item for sublist in row for item in sublist]
        keys[index] = set(flat_list)
        count[index] = len(set(flat_list))

    return keys, count


def calculate_mask(args):
    """Calculates the average of all masks
    """
    input_ = args["input"]
    state_ = args["state"]
    input_dir = state_["baseDirectory"]
    cache_dir = state_["cacheDirectory"]
    output_dir = state_["transferDirectory"]

    site_ids = input_.keys()
    avg_of_all = sum([
        nib.load(os.path.join(input_dir, site,
                              input_[site]['avg_nifti'])).get_fdata()
        for site in input_
    ]) / len(site_ids)

    # Threshold binarizer
    user_id = list(input_)[0]
    threshold = input_[user_id]["threshold"]

    mask_info = avg_of_all > threshold

    principal_image = nib.load(
        os.path.join(input_dir, user_id, input_[user_id]['avg_nifti']))
    header = principal_image.header
    affine = principal_image.affine

    clipped_img = nib.Nifti1Image(mask_info, affine, header)

    # TODO: Resampling (check with Eswar)
    voxel_size = input_[user_id]["voxel_size"]
    file = 'MNI152_T1_' + str(voxel_size) + 'mm_brain.nii'
    mni_image = os.path.join('/computation/templates', file)
    # I don't like these above 3 lines of code

    clipped_img = resample_to_img(clipped_img,
                                  mni_image,
                                  interpolation='nearest')

    output_file1 = os.path.join(output_dir, 'mask.nii')
    output_file2 = os.path.join(cache_dir, 'mask.nii')

    nib.save(clipped_img, output_file1)
    nib.save(clipped_img, output_file2)


def remote_0(args):
    """The first function in the remote computation chain
    """
    calculate_mask(args)
    input_ = args["input"]
    site_info = {
        site: input_[site]['categorical_dict']
        for site in input_.keys()
    }

    df = pd.DataFrame.from_dict(site_info)
    covar_keys, unique_count = return_uniques_and_counts(df)

    computation_output_dict = {
        "output": {
            "covar_keys": covar_keys,
            "global_unique_count": unique_count,
            "mask": 'mask.nii',
            "computation_phase": "remote_0"
        },
        "cache": {}
    }

    return json.dumps(computation_output_dict)


def remote_1(args):
    """ The second function in the local computation chain
    """
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

    beta_vector_0 = [
        np.array(input_list[site]["XtransposeX_local"]) for site in input_list
    ]

    beta_vector_1 = sum(beta_vector_0)

    all_lambdas = [input_list[site]["lambda"] for site in input_list]

    if np.unique(all_lambdas).shape[0] != 1:
        raise Exception("Unequal lambdas at local sites")

    beta_vector_1 = beta_vector_1 + np.unique(all_lambdas) * np.eye(
        beta_vector_1.shape[0])

    avg_beta_vector = np.matrix.transpose(
        sum([
            np.matmul(sp.linalg.inv(beta_vector_1),
                      input_list[site]["Xtransposey_local"])
            for site in input_list
        ]))

    mean_y_local = [input_list[site]["mean_y_local"] for site in input_list]
    count_y_local = [
        np.array(input_list[site]["count_local"]) for site in input_list
    ]
    mean_y_global = np.array(mean_y_local) * np.array(count_y_local)
    mean_y_global = np.sum(mean_y_global, axis=0) / np.sum(count_y_local,
                                                           axis=0)

    dof_global = sum(count_y_local) - avg_beta_vector.shape[1]

    output_dict = {
        "avg_beta_vector": avg_beta_vector.tolist(),
        "mean_y_global": mean_y_global.tolist(),
        "computation_phase": "remote_1"
    }

    cache_dict = {
        "avg_beta_vector": avg_beta_vector.tolist(),
        "mean_y_global": mean_y_global.tolist(),
        "dof_global": dof_global.tolist(),
        "X_labels": X_labels,
        "local_stats_dict": all_local_stats_dicts
    }

    computation_output_dict = {"output": output_dict, "cache": cache_dict}

    file_name = os.path.join(cache_dir, 'remote_cache')
    with open(file_name, 'w') as file_h:
        input_list[site] = json.dump(cache_dict, file_h)

    return json.dumps(computation_output_dict)


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

    input_list = dict()
    site_list = args["input"].keys()
    for site in site_list:
        file_name = os.path.join(input_dir, site, OUTPUT_FROM_LOCAL)
        with open(file_name, 'r') as file_h:
            input_list[site] = json.load(file_h)

    cache_list = read_file(args, "cache", 'remote_cache')

    X_labels = args["cache"]["X_labels"]

    all_local_stats_dicts = cache_["local_stats_dict"]

    avg_beta_vector = cache_list["avg_beta_vector"]
    dof_global = cache_list["dof_global"]

    SSE_global = sum(
        [np.array(input_list[site]["SSE_local"]) for site in input_list])
    #    SST_global = sum(
    #        [np.array(input_list[site]["SST_local"]) for site in input_list])
    varX_matrix_global = sum([
        np.array(input_list[site]["varX_matrix_local"]) for site in input_list
    ])

    #    r_squared_global = 1 - (SSE_global / SST_global)
    MSE = SSE_global / np.array(dof_global)

    ts_global = []
    ps_global = []

    for i, _ in enumerate(MSE):
        var_covar_beta_global = MSE[i] * sp.linalg.inv(varX_matrix_global)
        se_beta_global = np.sqrt(var_covar_beta_global.diagonal())
        ts = (avg_beta_vector[i] / se_beta_global).tolist()
        ps = reg.t_to_p(ts, dof_global[i])
        ts_global.append(ts)
        ps_global.append(ps)

    print_pvals(args, ps_global, ts_global, X_labels)
    print_beta_images(args, avg_beta_vector, X_labels)

    # Block of code to print local stats as well
    sites = [site for site in input_list]

    all_local_stats_dicts = dict(zip(sites, all_local_stats_dicts))

    # Block of code to print just global stats
    global_dict_list = encode_png(args)

    # Print Everything
    keys2 = ["global_stats", "local_stats"]
    output_dict = dict(zip(keys2, [global_dict_list, all_local_stats_dicts]))

    computation_output_dict = {"output": output_dict, "success": True}

    return json.dumps(computation_output_dict)


if __name__ == '__main__':

    PARSED_ARGS = json.loads(sys.stdin.read())
    PHASE_KEY = list(reg.list_recursive(PARSED_ARGS, 'computation_phase'))

    if "local_0" in PHASE_KEY:
        sys.stdout.write(remote_0(PARSED_ARGS))
    elif "local_1" in PHASE_KEY:
        sys.stdout.write(remote_1(PARSED_ARGS))
    elif "local_2" in PHASE_KEY:
        sys.stdout.write(remote_2(PARSED_ARGS))
    else:
        raise ValueError("Error occurred at Remote")
