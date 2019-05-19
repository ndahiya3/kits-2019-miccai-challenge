#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 22:54:10 2019
Script to run inference on Kidney data. The model is trained on cropped frames
as kidney and tumor are very small compared to image size.
Runs inference using a trained model on a batch of testing sets.
Saves all predicted masks in '*_pred_mask.nrrd' format. In case of multiclass,
labels are compressed to single file with 0 as background and 1,2,...,nb_classes
as other class labels.

Reads the test set id's from text file. Model to run's name is usually based
on experiment name.
@author: ndahiya
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:07:33 2019


@author: ndahiya
"""
import argparse
from os import path
import numpy as np
import tensorflow as tf
import SimpleITK as sitk

import sys
sys.path.insert(0, path.abspath('..'))
import helpers.utilities as utils
from models.unet_model_dilated_conv import unet_model_dilated_conv
from models.unet_model_deeper_dilated_conv import unet_model_deeper_dilated_conv

def crop_dataset(dataset):
  """
  Crop a kidney frame. Crop region is 256x256 around image center.
  """
  #cropped = frame[256-192:256+192,256-192:256+192]

  frames = dataset.shape[0]
  cropped_dataset = np.zeros((frames,256,256))
  for frame in range(frames):
    img = dataset[frame]
    cropped_dataset[frame] = img[256-128+25:256+128+25,256-128:256+128]
  #print(cropped.shape)
  return cropped_dataset

# Use this if running with default arguments
exp_name_default="unet_tversky_full_cropped"

parser = argparse.ArgumentParser(description="This script runs inference using "
                                 "a pretrained model on a set of "
                                 "datasets specified in text file.")
parser.add_argument("--curr_exp_name", help="Current experiment name.",
                    type=str)
parser.add_argument("--out_dir", help="Output directory to save inference masks.",
                    type=str)
parser.add_argument("--test_ids_file", help="Text file containing ids of test "
                    "datasets to run.", type=str)
parser.add_argument("--num_dsets", help="Number of test datasets to run.",
                    type=str)
parser.add_argument("--test_data_location", help="Directory containing test data.",
                    type=str)
parser.add_argument("--model_name", help="Full name/path of trained model.",
                    type=str)
parser.add_argument("--nb_classes", help="Number of classes in the model.",
                    type=int)
parser.add_argument("--batch_size", help="Batch size: reduce for smaller gpu.",
                    type=int) # it's more efficient if batch_size evenly divides
                              # num slices/dataset
parser.add_argument("--gpu_device", help="GPU to use e.g. /gpu:0 or /cpu:0",
                    type=str)

parser.set_defaults(curr_exp_name=exp_name_default,
                    out_dir=exp_name_default+'/',
                    test_ids_file="../resources/test_ids.txt",
                    num_dsets=74,
                    test_data_location="../resources/training_data/test/",
                    model_name='models/unet_' + exp_name_default + '.hdf5',
                    num_classes=3,
                    batch_size=160,
                    gpu_device='/gpu:0')
args = parser.parse_args()

curr_exp_name       = args.curr_exp_name
out_folder          = args.out_dir
test_ids_file       = args.test_ids_file
nb_dsets_to_run     = args.num_dsets
test_data_location  = args.test_data_location
model_name          = args.model_name
nb_classes          = args.num_classes
batch_size          = args.batch_size
device              = args.gpu_device

# Convert paths to absolute paths
test_ids_file = path.abspath(test_ids_file)
test_list = utils.get_dataset_ids(test_ids_file, nb_dsets_to_run)
#test_list = test_list[:1]
nb_dsets_to_run = len(test_list) # In case ids_file had less than we wanted to run

model_name = path.abspath(model_name)

with tf.device(device):
  #model = unet_model_tack_arch(model_name, num_classes=nb_classes)
  #model = unet_model_dilated_conv(model_name, num_classes=nb_classes)
  model = unet_model_dilated_conv(model_name, num_classes=nb_classes, input_size=(1,256,256))
  #model.summary()

  # Predict and save all test datasets
  for idx, test_id in enumerate(test_list):
    print('Processing {}: {} of {}'.format(test_id, idx+1, nb_dsets_to_run))
    curr_test_dicom_name = test_data_location + test_id + '_dicom.nrrd'
    curr_test_dicom_name = path.abspath(curr_test_dicom_name)

    dicom = sitk.ReadImage(curr_test_dicom_name)
    dicom_arr = sitk.GetArrayFromImage(dicom)

    # Change needed for Kidney Data shape [512, 512, frames] ==> [frames, 512, 512]
    dicom_arr = np.moveaxis(dicom_arr, -1, 0)
    # Transpose each frame
    num_frames = dicom_arr.shape[0]
    for idx in range(num_frames):
      dicom_arr[idx] = dicom_arr[idx].T

    dicom_shape = dicom_arr.shape
    if dicom_shape[1] != 512 or dicom_shape[2] != 512:
      print('Dimensions mismatch for {}'.format(test_id))
      continue
    min_val = dicom_arr.min()
    if min_val < 0:
      dicom_arr = dicom_arr - min_val # Shift data to make min == 0
    dicom_arr = crop_dataset(dicom_arr)
    dicom_shape = dicom_arr.shape
    out_mask  = np.zeros(dicom_shape, np.uint8)

    inference_req_shape = (dicom_shape[0],1,dicom_shape[1], dicom_shape[2])
    dicom_arr = np.reshape(dicom_arr, inference_req_shape) # Hardcoded input shape
    if dicom_shape[0] < 160:
      batch_size = dicom_shape[0]
    else:
      batch_size = 160
#    if batch_size > dicom_shape[0]:
#      batch_size = dicom_shape[0]
    # Run prediction on batch_size slices at a time
    for idx in range(0, dicom_shape[0], batch_size):
      start_idx = idx
      end_idx   = idx + batch_size
      result = model.predict(dicom_arr[start_idx:end_idx,...], verbose=1)
      if nb_classes == 1:
        pred_labels = np.squeeze(result, axis=1)
        #print(pred_labels.max())
        #print(pred_labels.min())
        pred_labels[np.where(pred_labels > 0.1)] = 1
        pred_labels = pred_labels.astype(np.uint8)
        #print(np.count_nonzero(pred_labels))
      else:
        pred_labels = result.argmax(axis=1)
        pred_labels = pred_labels.astype(np.uint8)
      # Copy predicted slices
      out_mask[start_idx:end_idx] = pred_labels

    out_file_name = out_folder + test_id + '_pred_mask.nrrd'
    out_file_name = path.abspath(out_file_name)
    sitk.WriteImage(sitk.GetImageFromArray(out_mask), out_file_name, False)
    # Write cropped DICOM array as well
    out_file_name = out_folder + test_id + '_dicom.nrrd'
    out_file_name = path.abspath(out_file_name)
    dicom_arr = np.squeeze(dicom_arr, axis=1)
    sitk.WriteImage(sitk.GetImageFromArray(dicom_arr), out_file_name, False)
print('Done')

