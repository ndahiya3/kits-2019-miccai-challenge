#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:58:38 2019
  Save data into train/valid/test folders.
  Different script splits all available training dataset names into
  train/test/valid. This script copies the datasets into separate folders
  for train/test/valid. This structure of directories is needed by Deep Learning
  libraries.
@author: ndahiya, aloksh
"""
import os
import shutil
import SimpleITK as sitk

def saveData(dset_id_list, fold_type):
  # Load DICOM and masks

  # Alok Directory paths
  #in_base_dir  = '/home/alok/projects/kits19/data'
  #out_base_dir = '/home/alok/projects/kits-2019-miccai-challenge/resources/training_data'

  # Navdeep Directory paths
  in_base_dir  = '/home/ndahiya/school/MICCAI-Challanges/kits-2019-kidney-challenge/kits19/data'
  out_base_dir = '/home/ndahiya/school/MICCAI-Challanges/kits-2019-kidney-challenge/kits-2019-miccai-challenge/resources/training_data'

  for dset_id in dset_id_list:
    # Read dicom and mask image

    data_dir = os.path.join(in_base_dir, dset_id)
    dicom_name = data_dir + '/imaging.nii.gz'
    mask_name  = data_dir + '/segmentation.nii.gz'
    print (dicom_name)
    print(mask_name)

    out_dicom_name = os.path.join(out_base_dir, fold_type, dset_id + '_dicom.nii.gz')
    out_mask_name  = os.path.join(out_base_dir, fold_type, dset_id + '_seg_mask.nii.gz')

    # Save dicom and mask into corresponding folder.
    shutil.copy(dicom_name, out_dicom_name)
    shutil.copy(mask_name, out_mask_name)

  print ('Done writing ' + fold_type + ' data')

# Read train/valid/test split files
train_list = []
valid_list = []
test_list  = []

with open('train_ids.txt', 'r') as f:
    for line in f:
        train_list.append(line.strip())

with open('valid_ids.txt', 'r') as f:
    for line in f:
        valid_list.append(line.strip())

with open('test_ids.txt', 'r') as f:
    for line in f:
        test_list.append(line.strip())

print('Images in train: ', len(train_list))
print('Images in valid: ', len(valid_list))
print('Images in test : ', len(test_list))

# Write images
saveData(train_list, 'train')
saveData(valid_list, 'valid')
saveData(test_list, 'test')

print('All done!!')