#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 05:09:13 2019

@author: Harshvardhan
"""
import base64
import nibabel as nib
import numpy as np
import os
import pandas as pd
import sys
import traceback
from nilearn import plotting
from nipype.interfaces import afni

np.seterr(divide = 'ignore')

MASK = os.path.join('/computation', 'mask_4mm.nii')


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def encode_png(args):
    # Begin code to serialize png images
    png_files = sorted(os.listdir(args["state"]["outputDirectory"]))

    encoded_png_files = []
    for file in png_files:
        if file.endswith('.png'):
            mrn_image = os.path.join(args["state"]["outputDirectory"], file)
            with open(mrn_image, "rb") as imageFile:
                mrn_image_str = base64.b64encode(imageFile.read())
            encoded_png_files.append(mrn_image_str)

    return dict(zip([file for file in png_files if file.endswith('.png')], encoded_png_files))


def print_beta_images(args, avg_beta_vector, X_labels):
    beta_df = pd.DataFrame(avg_beta_vector, columns=X_labels)

    images_folder = args["state"]["outputDirectory"]

    mask = nib.load(MASK)

    for column in beta_df.columns:
        new_data = np.zeros(mask.shape)
        new_data[mask.get_data() > 0] = beta_df[column]

        image_string = 'beta_' + str(column)

        clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
        output_file = os.path.join(images_folder, image_string)

        nib.save(clipped_img, output_file + '.nii')

        plotting.plot_stat_map(
            clipped_img,
            output_file=output_file,
            display_mode='ortho',
            colorbar=True)


def print_pvals(args, ps_global, ts_global, X_labels):
    p_df = pd.DataFrame(ps_global, columns=X_labels)
    t_df = pd.DataFrame(ts_global, columns=X_labels)

    # TODO manual entry, remove later
    images_folder = args["state"]["outputDirectory"]

    mask = nib.load(MASK)

    for column in p_df.columns:
        new_data = np.zeros(mask.shape)
        new_data[mask.get_data() >
                 0] = -1 * np.log10(p_df[column]) * np.sign(t_df[column])

        image_string = 'pval_' + str(column)

        clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
        output_file = os.path.join(images_folder, image_string)

        nib.save(clipped_img, output_file + '.nii')

        #        thresholdh = max(np.abs(p_df[column]))
        plotting.plot_stat_map(
            clipped_img,
            output_file=output_file,
            display_mode='ortho',
            colorbar=True)


def resample_nifti_images(image_file, resampled_file, voxel_dimensions, resample_method):
    """Resample the NIfTI images in a folder and put them in a new folder
    Args:
        images_location: Path where the images are stored
        voxel_dimension: tuple (dx, dy, dz)
        resample_method: NN - Nearest neighbor
                         Li - Linear interpolation
    Returns:
        None:
    """
    resample = afni.Resample()
    resample.inputs.environ = {'AFNI_NIFTI_TYPE_WARN': 'NO'}
    resample.inputs.in_file = image_file
    resample.inputs.out_file = resampled_file
    resample.inputs.voxel_size = voxel_dimensions
    resample.inputs.outputtype = 'NIFTI'
    resample.inputs.resample_mode = resample_method
    resample.run()
