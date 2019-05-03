#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:26:01 2019
Extract individual frames from specific DICOMs and corresponding masks.
Used to generate the basic data for training/testing and/or validation.

@author: ndahiya, aloksh
"""

import os
import argparse
import subprocess
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(1, os.path.abspath('..'))
from helpers.generate_sample_training_data import extract_data

parser = argparse.ArgumentParser(description="This program extracts individual slices for model training.")

parser.add_argument("--out_dir", help="Base directory to extract slices to.",
                    type=str)
parser.add_argument("--train_dsets_num", help="Number of training datasets to extract.",
                    type=int)
parser.add_argument("--valid_dsets_num", help="Number of validation datasets to extract.",
                    type=int)
parser.add_argument("--classes_to_keep", help="Class labels to keep [combined or separate].",
                    type=int, nargs='*')
parser.add_argument("--convert_masks_to_binary", help="Combine Class labels to keep to one foreground mask?",
                    action='store_true')
parser.add_argument("--train_ids_file", help="Text file with list of train datasets OAI ids.",
                    type=str)
parser.add_argument("--valid_ids_file", help="Text file with list of validation datasets OAI ids.",
                    type=str)
parser.add_argument("--train_in_dir", help="Directory location of training datasets.",
                    type=str)
parser.add_argument("--valid_in_dir", help="Directory location of validation datasets.",
                    type=str)
###### ------ Use following to set defaults -------------- ######

# Max training datasets   = 350
# Max validation datasets = 100
# Max testing datasets    = 251
# Total = 701

# Class labels


parser.set_defaults(out_dir='unet_tversky_1',
                    train_dsets_num=1,
                    valid_dsets_num=1,
                    classes_to_keep=[1,2],
                    train_ids_file='train_ids.txt',
                    valid_ids_file='valid_ids.txt',
                    train_in_dir='training_data/train',
                    valid_in_dir='training_data/valid')
args = parser.parse_args()

base_out_dir = args.out_dir
train_dsets_to_extract  = args.train_dsets_num
valid_dsets_to_extract  = args.valid_dsets_num
classes_to_keep         = args.classes_to_keep
train_ids_file          = args.train_ids_file
valid_ids_file          = args.valid_ids_file
train_in_dir            = args.train_in_dir
valid_in_dir            = args.valid_in_dir
convert_masks_to_binary = args.convert_masks_to_binary

###### ------TO DO: Add these to argument parser with defaults ----- ######
extract_only_with_foreground = False

val = subprocess.check_call("./create_train_dir_structure '%s'" % base_out_dir, shell=True)
train_out_dicom_dir = os.path.join(base_out_dir, 'train/dicoms')
train_out_masks_dir = os.path.join(base_out_dir, 'train/masks')
valid_out_dicom_dir = os.path.join(base_out_dir, 'valid/dicoms')
valid_out_masks_dir = os.path.join(base_out_dir, 'valid/masks')

# Convert paths to absolute
train_ids_file      = os.path.abspath(train_ids_file)
train_in_dir        = os.path.abspath(train_in_dir)
train_out_dicom_dir = os.path.abspath(train_out_dicom_dir)
train_out_masks_dir = os.path.abspath(train_out_masks_dir)
valid_ids_file      = os.path.abspath(valid_ids_file)
valid_in_dir        = os.path.abspath(valid_in_dir)
valid_out_dicom_dir = os.path.abspath(valid_out_dicom_dir)
valid_out_masks_dir = os.path.abspath(valid_out_masks_dir)


# Extract training data
extract_data(train_ids_file, train_in_dir, train_out_dicom_dir,
             train_out_masks_dir, train_dsets_to_extract,
             convert_masks_to_binary, classes_to_keep,
             extract_only_with_foreground)

# Extract validation data
extract_data(valid_ids_file, valid_in_dir, valid_out_dicom_dir,
             valid_out_masks_dir, valid_dsets_to_extract,
             convert_masks_to_binary, classes_to_keep,
             extract_only_with_foreground)
