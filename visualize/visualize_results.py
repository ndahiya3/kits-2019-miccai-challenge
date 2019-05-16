#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:18:09 2019

Visualize the inference masks, overlaid over true dicom mri as well as ground
truth masks.

@author: ndahiya
"""

import numpy as np
import SimpleITK as sitk
from visualizer import multi_slice_overlay_viewer
from visualizer import multi_slice_viewer

def transpose(img_arr):
  for idx in range(img_arr.shape[0]):
    img_arr[idx] = img_arr[idx].T
  return img_arr

dset_id = 'case_00017'
#dset_id = 'VSEG_SET_1_00131352-1'#_seg_mask
dicom_dir = '../resources/training_data/test/'
#dicom_dir = '../resources/sn_training_data/'
dicom_suffix = '_dicom.nrrd'

pred_mask_dir = '../results/tversky_full_lr_pt4/'
pred_mask_suffix = '_pred_mask.nrrd'
#pred_mask_suffix = '_clean_mask.nrrd'

true_mask_dir = '../resources/training_data/test/'
#true_mask_dir = '../resources/sn_training_data/'
true_mask_suffix = '_seg_mask.nrrd'

dicom_path     = dicom_dir + dset_id + dicom_suffix
pred_mask_path = pred_mask_dir + dset_id + pred_mask_suffix
true_mask_path = true_mask_dir + dset_id + true_mask_suffix

dicom_itk = sitk.ReadImage(dicom_path)
dicom_arr = sitk.GetArrayFromImage(dicom_itk)
dicom_arr = np.moveaxis(dicom_arr, -1, 0)
dicom_arr = transpose(dicom_arr)
print(dicom_arr.shape)

pred_mask_itk = sitk.ReadImage(pred_mask_path)
pred_mask_arr = sitk.GetArrayFromImage(pred_mask_itk)
pred_mask_arr = transpose(pred_mask_arr)
print(pred_mask_arr.shape)

true_mask_itk = sitk.ReadImage(true_mask_path)
true_mask_arr = sitk.GetArrayFromImage(true_mask_itk)
true_mask_arr = np.moveaxis(true_mask_arr, -1, 0)
true_mask_arr = transpose(true_mask_arr)
print(true_mask_arr.shape)

multi_slice_overlay_viewer(dicom_arr, true_mask_arr, pred_mask_arr, dset_id)
#multi_slice_viewer(dicom_arr)