#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 21:01:15 2019

@author: hgazula
"""
import os

import nibabel as nib
import numpy as np
from nilearn.image import resample_img, resample_to_img

from parsers import parse_for_covar_info

MASK = 'mask.nii'
MNI_TEMPLATE = '/computation/templates/MNI152_T1_1mm_brain.nii'


# TODO: Check if I can do away with the try and except block
def nifti_to_data(args, X):
    """Read nifti files as matrices
    """
    voxel_size = args["input"]["voxel_size"]
    try:
        mask_data = nib.load(os.path.join(args["state"]["baseDirectory"],
                                          MASK)).get_fdata()
    except FileNotFoundError:
        raise Exception("Missing Mask at " + args["state"]["clientId"])

    mni_image = os.path.join(args["state"]["baseDirectory"],
                             'mni_downsampled.nii')
    appended_data = []
    for image in X.index:
        input_file = os.path.join(args["state"]["baseDirectory"], image)
        try:
            if nib.load(input_file).header.get_zooms()[0] == voxel_size:
                image_data = nib.load(input_file).get_data()
            else:
                clipped_img = resample_to_img(input_file, mni_image)
                image_data = clipped_img.get_data()

            if np.all(np.isnan(image_data)) or np.count_nonzero(
                    image_data) == 0 or image_data.size == 0:
                X.drop(index=image, inplace=True)
                continue
            else:
                appended_data.append(image_data[mask_data > 0])
        except FileNotFoundError:
            continue

    y = np.vstack(appended_data)
    y = y.astype('float64')

    return X, y


def average_nifti(args):
    """Reads in all the nifti images and calculates their average
    """
    state_ = args["state"]
    input_dir = state_["baseDirectory"]
    output_dir = state_["transferDirectory"]

    covar_x, _ = parse_for_covar_info(args)

    appended_data = 0
    for image in covar_x.index:
        try:
            image_data = nib.load(os.path.join(input_dir, image)).get_fdata()
            if np.all(np.isnan(image_data)) or np.count_nonzero(
                    image_data) == 0 or image_data.size == 0:
                covar_x.drop(index=image, inplace=True)
                continue
            else:
                appended_data += image_data
        except FileNotFoundError:
            covar_x.drop(index=image, inplace=True)
            continue

    sample_image = nib.load(os.path.join(input_dir, covar_x.index[0]))
    header = sample_image.header
    affine = sample_image.affine

    avg_nifti = appended_data / len(covar_x.index)

    clipped_img = nib.Nifti1Image(avg_nifti, affine, header)
    output_file = os.path.join(output_dir, 'avg_nifti.nii')
    nib.save(clipped_img, output_file)


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
    voxel_size = input_[user_id]["voxel_size"]

    mask_info = avg_of_all > threshold

    principal_image = nib.load(
        os.path.join(input_dir, user_id, input_[user_id]['avg_nifti']))
    header = principal_image.header
    affine = principal_image.affine

    clipped_img = nib.Nifti1Image(mask_info, affine, header)
    mni_image = os.path.join('/computation/templates', MNI_TEMPLATE)

    reoriented_mni = resample_to_img(mni_image,
                                     clipped_img,
                                     interpolation='linear')
    downsampled_mni = resample_img(reoriented_mni,
                                   target_affine=np.eye(3) * voxel_size,
                                   interpolation='linear')

    downsampled_mask = resample_to_img(clipped_img,
                                       downsampled_mni,
                                       interpolation='nearest')

    output_file1 = os.path.join(output_dir, 'mask.nii')
    output_file2 = os.path.join(cache_dir, 'mask.nii')
    output_file3 = os.path.join(output_dir, 'mni_downsampled.nii')
    output_file4 = os.path.join(cache_dir, 'mni_downsampled.nii')

    nib.save(downsampled_mask, output_file1)
    nib.save(downsampled_mask, output_file2)
    nib.save(downsampled_mni, output_file3)
    nib.save(downsampled_mni, output_file4)
