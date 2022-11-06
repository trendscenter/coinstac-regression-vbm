#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 05:09:13 2019

@author: Harshvardhan
"""
import base64
import os

import numpy as np
import pandas as pd

import nibabel as nib
from nilearn import plotting

np.seterr(divide="ignore")

MASK = "mask.nii"


def saveBin(path, arr):
    with open(path, "wb+") as fh:
        header = "%s" % str(arr.dtype)
        for index in arr.shape:
            header += " %d" % index
        header += "\n"
        fh.write(header.encode())
        fh.write(arr.data.tobytes())
        os.fsync(fh)


def loadBin(path):
    with open(path, "rb") as fh:
        header = fh.readline().decode().split()
        dtype = header.pop(0)
        arrayDimensions = []
        for dimension in header:
            arrayDimensions.append(int(dimension))
        arrayDimensions = tuple(arrayDimensions)
        return np.frombuffer(fh.read(), dtype=dtype).reshape(arrayDimensions)


def encode_png(args):
    """Serialize png images."""
    png_files = sorted(os.listdir(args["state"]["outputDirectory"]))

    encoded_png_files = []
    for file in png_files:
        if file.endswith(".png"):
            mrn_image = os.path.join(args["state"]["outputDirectory"], file)
            with open(mrn_image, "rb") as image_file:
                mrn_image_str = base64.b64encode(image_file.read())
            encoded_png_files.append(mrn_image_str)

    return dict(zip([f for f in png_files if f.endswith(".png")], encoded_png_files))


def print_beta_images(args, avg_beta_vector, covar_labels):
    """Print regression coefficients as nifti files."""
    beta_df = pd.DataFrame(avg_beta_vector, columns=covar_labels)

    state_ = args["state"]
    images_folder = state_["outputDirectory"]

    try:
        mask = nib.load(os.path.join(args["state"]["baseDirectory"], MASK))
    except FileNotFoundError:
        mask = nib.load(os.path.join(args["state"]["cacheDirectory"], MASK))

    for column in beta_df:
        new_data = np.zeros(mask.shape)
        new_data[mask.get_data() > 0] = beta_df[column]

        image_string = "beta_" + str(column)

        clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
        output_file = os.path.join(images_folder, image_string)

        nib.save(clipped_img, output_file + ".nii")

        plotting.plot_stat_map(
            clipped_img, output_file=output_file, display_mode="ortho", colorbar=True
        )


def print_pvals(args, ps_global, ts_global, covar_labels):
    """Print regression coefficients' p-values as nifti files."""
    p_df = pd.DataFrame(ps_global, columns=covar_labels)
    t_df = pd.DataFrame(ts_global, columns=covar_labels)

    state_ = args["state"]
    images_folder = state_["outputDirectory"]

    try:
        mask = nib.load(os.path.join(args["state"]["baseDirectory"], MASK))
    except FileNotFoundError:
        mask = nib.load(os.path.join(args["state"]["cacheDirectory"], MASK))

    for column in p_df:
        new_data = np.zeros(mask.shape)
        new_data[mask.get_data() > 0] = (
            -1 * np.log10(p_df[column]) * np.sign(t_df[column])
        )

        image_string = "pval_" + str(column)

        clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
        output_file = os.path.join(images_folder, image_string)

        nib.save(clipped_img, output_file + ".nii")

        plotting.plot_stat_map(
            clipped_img, output_file=output_file, display_mode="ortho", colorbar=True
        )


def print_r2_image(args, beta_df):
    """Print regression coefficients as nifti files."""
    state_ = args["state"]
    images_folder = state_["outputDirectory"]

    try:
        mask = nib.load(os.path.join(args["state"]["baseDirectory"], MASK))
    except FileNotFoundError:
        mask = nib.load(os.path.join(args["state"]["cacheDirectory"], MASK))

    new_data = np.zeros(mask.shape)
    new_data[mask.get_data() > 0] = beta_df

    image_string = "r_squared"

    clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
    output_file = os.path.join(images_folder, image_string)

    nib.save(clipped_img, output_file + ".nii")

    plotting.plot_stat_map(
        clipped_img, output_file=output_file, display_mode="ortho", colorbar=True
    )
