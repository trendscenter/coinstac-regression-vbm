#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 05:09:13 2019

@author: Harshvardhan
"""
import nibabel as nib
import numpy as np
import os
import pandas as pd
from nilearn import plotting

MASK = os.path.join('/computation', 'mask_4mm.nii')


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