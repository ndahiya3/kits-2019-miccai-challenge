#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:41:45 2019
Main script to define training experiment parameters including train/validation
data locations etc. and run training for Smith and Nephew Data.
@author: ndahiya
"""

import os
from train import train
import argparse
parser = argparse.ArgumentParser(description="This script runs training with specified settings.")
parser.add_argument("--curr_exp_name", help="Name of current experiment.",
                    type=str)
parser.add_argument("--batch_size", help="Training batch size.",
                    type=int)
parser.add_argument("--num_epochs", help="Number of training epochs to run.",
                    type=int)
parser.add_argument("--num_classes", help="Number of object classes.",
                    type=int)

parser.set_defaults(curr_exp_name='unet_tversky_mini',
                    batch_size=8,
                    num_epochs=50,
                    num_classes=3)

args = parser.parse_args()

curr_experiment_name = args.curr_exp_name
batch_size           = args.batch_size
num_epochs           = args.num_epochs
num_classes          = args.num_classes
train_3d             = False
train_images_dir = curr_experiment_name + '/train'
valid_images_dir = curr_experiment_name + '/valid'
save_aug_images_dir = None

model_save_name = 'unet_' + curr_experiment_name + '.hdf5'
pretrained_checkpoint = None #'unet_knee_sn_3d_unet_test_tversky.hdf5'
curr_log_dir = 'logs/' + curr_experiment_name
csv_log_file = curr_experiment_name + '.csv'
device = '/gpu:0'

# Resolve paths before passing to train.py
train_images_dir = os.path.abspath(train_images_dir)
valid_images_dir = os.path.abspath(valid_images_dir)
if save_aug_images_dir is not None:
  save_aug_images_dir = os.path.abspath(save_aug_images_dir)
model_save_name = os.path.abspath(model_save_name)
if pretrained_checkpoint is not None:
  pretrained_checkpoint = os.path.abspath(pretrained_checkpoint)
curr_log_dir = os.path.abspath(curr_log_dir)
csv_log_file = os.path.abspath(csv_log_file)


train(train_images_dir, valid_images_dir, save_aug_images_dir,
      batch_size, model_save_name, pretrained_checkpoint, curr_log_dir,
      csv_log_file, num_classes, num_epochs, device, train_3d)

print("Finished Experiment {}.".format(curr_experiment_name))