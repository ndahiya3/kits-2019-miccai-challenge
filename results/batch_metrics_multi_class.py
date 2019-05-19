#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:29:58 2019

For Smith and Nephew Data.
Calculate and save/print metrics for comparing a whole batch of
manual vs automatic segmentations for the multi class case. Each anatomy
has a unique integer label.

@author: ndahiya
"""

import numpy as np
import SimpleITK as sitk
from os import path
import sys
sys.path.insert(0, path.abspath('..'))
from helpers.calc_metrics import dice_coef_mc
from helpers.calc_metrics import mc_calc_localized_metrics
import helpers.utilities as utils
  
# --------------- Main Code -----------------------------------
localized_metrics = False
curr_exp_name = 'sn_mc_tversky_deeper_701_processed'

test_ids_file = '../resources/test_ids.txt'
true_mask_location = '../resources/training_data/test/'
pred_mask_location = curr_exp_name + '/'

true_mask_suffix = '_seg_mask.nrrd'
#pred_mask_suffix = '_pred_mask.nrrd'
pred_mask_suffix = '_clean_mask.nrrd'

#metrics_out_file = curr_exp_name + '_clean_metrics.txt'
metrics_out_file = curr_exp_name + '_metrics.txt'
#metrics_out_file = curr_exp_name + '_localized_metrics.txt'

# Class labels
# 0 - background
# 1 - femoral bone + cartilage
# 2 - tibial bone + cartilage

label_dict = {
    0: "Whole Knee",
    1: "Femur Combined",
    2: "Tibia Combined"
    }
classes_present = []
classes_present.append(1) # Femoral bone + cartilage
classes_present.append(2) # Tibial bone + cartilage

nb_classes = len(classes_present)

test_ids_file = path.abspath(test_ids_file)
test_list = utils.get_dataset_ids(test_ids_file, 300)
#test_list = utils.get_dataset_ids(test_ids_file, 3)
nb_dsets_to_run = len(test_list)

# Get metrics for the list of test datasets 
metrics_out_file = path.abspath(metrics_out_file)
metrics_file = open(metrics_out_file, 'w')

dice_scores = np.zeros((nb_dsets_to_run, nb_classes), dtype=np.float32)

for idx, test_id in enumerate(test_list):
  true_mask_file_name = true_mask_location + test_id + true_mask_suffix
  true_mask_file_name = path.abspath(true_mask_file_name)
  predicted_mask_name = pred_mask_location + test_id + pred_mask_suffix
  predicted_mask_name = path.abspath(predicted_mask_name)
  
  true_mask_itk = sitk.ReadImage(true_mask_file_name)
  pred_mask_itk = sitk.ReadImage(predicted_mask_name)
  true_mask = sitk.GetArrayFromImage(true_mask_itk)
  pred_mask = sitk.GetArrayFromImage(pred_mask_itk)
  
  if localized_metrics is True:
    dice_scores_list, _ = mc_calc_localized_metrics(true_mask, pred_mask)
    dice_scores[idx] = np.asarray(dice_scores_list[1:])
  else:
    dice_scores_list = dice_coef_mc(true_mask, pred_mask, nb_classes=nb_classes+1)
    dice_scores[idx] = np.asarray(dice_scores_list[1:]) # Ignore background
  
  utils.mc_print_dset_metrics(test_id, dice_scores_list, label_dict, classes_present)
  utils.mc_print_dset_metrics(test_id, dice_scores_list, label_dict, classes_present, 
                     out_file=metrics_file)

# Calculate and write aggregate statistics
utils.mc_calc_print_aggregate_metrics(test_list, dice_scores, label_dict,
                                 classes_present)
utils.mc_calc_print_aggregate_metrics(test_list, dice_scores, label_dict,
                                 classes_present, out_file=metrics_file)  
metrics_file.close()
