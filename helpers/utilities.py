#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 13:00:56 2019

Utility functions helpful for other codes.

@author: ndahiya
"""

import numpy as np
import SimpleITK as sitk
import os

def get_files_list_dir(dir_path):
  """
  Get the list of files present in a particular directory.
  # Arguments:
      dir_path: input directory name/path.
  # Returns:
      List of files present in the directory
  """
  dir_path = os.path.abspath(dir_path)
  file_list = []
  for root, dirs, files in os.walk(dir_path):
    for name in files:
      file_list.append(os.path.join(root, name))
  
  return file_list

def read_arr_from_itk(file_name):
  """
  Read dataset file using SimpleITK and return corresponding numpy array
  in float32 format
  # Arguments:
      file_name: String, name of input file
  # Returns:
      Image data in float32 numpy array format
  """
  file_name = os.path.abspath(file_name)
  dset_itk  = sitk.ReadImage(file_name)
  dset_arr  = sitk.GetArrayFromImage(dset_itk)
  dset_arr  = dset_arr.astype(np.float32)
  
  return dset_arr

def get_dataset_ids(oai_ids_file, num_datasets):
  """
  Get dataset IDs from file which could be test, train, or validations
  file containing datasets' OAI IDs text file.
  
  Parameters
  ----------
  oai_ids_file : string
                 Filename containing dataset ids to extract
                 Expects absolute file path
  num_datasets : integer
                 Num datasets to return in list. If requested more than present
                 in input file return all present only
  """
  dset_list  = []
  with open(oai_ids_file, 'r') as f:
    for line in f:
      dset_list.append(line.strip())
  nb_found = len(dset_list)
  if num_datasets > nb_found:
    num_datasets = nb_found
  
  dset_list = dset_list[:num_datasets]
  
  return dset_list

def resampleITKLabel(image_itk, reference_itk):
  """
  Resample SimpleITK label mask image to a reference image.
  Since we are re-sampling label mask, using Nearest Neighbor interpolation.
  
  Parameters
  ----------
  image : ITK label mask image to re-sample
  reference : ITK image to use as a reference for image dimensions, orientation
              spacing, etc.
  """
  
  resampler = sitk.ResampleImageFilter()
  resampler.SetInterpolator(sitk.sitkNearestNeighbor)
  resampler.SetReferenceImage(reference_itk)
  return resampler.Execute(image_itk)

# ----------------- Functions for printing one class metrics -----------------
def oc_print_heading(metric_name_format_dict={}, out_file=None):
  """
  Function to print top level heading for one class metrics file.
  Dictionary with names of metrics with corresp. format specification
  for printing.
  If out_file = None, print to Stdout else print to file.
  """
  headings_list = ['Dataset', 'Anatomy']
  headings_format = '{:<10s}{:<20s}'
  for key in metric_name_format_dict:
    headings_list.append(key)
    headings_format = headings_format + '{:<10s}'
  print(headings_format.format(*headings_list), file=out_file)
  print('-'*100, file=out_file)

def oc_print_dset_score(test_id, anatomy_name, metric_name_format_dict={},
                            metrics=[], out_file=None):
  """
  Function to print metrics for a particular dataset.
  If out_file = None, prints to Stdout otherwise to specified file
  """
  metrics_format = '{:<10s}{:<20s}'
  for key, val in metric_name_format_dict.items():
    metrics_format = metrics_format + val
  
  print(metrics_format.format(test_id, anatomy_name, *metrics), file=out_file)
  
def oc_calc_print_aggregate(id_list, metric_name_format_dict={},
                                metrics=None, out_file=None):
  """
  Function to calculate and print aggregate scores for one class metrics.
  If out_file = None, prints to Stdout otherwie to specified file.
  Metrics is numpy array with shape = nb_dsets x nb_metrics
  """
  # Calculate and write aggregate statistics; VD can be pos/neg but we care
  # about abs otherwise pos/neg could cancel each other and make results
  # look better than they are
   
  nb_dsets   = metrics.shape[0]
  nb_metrics = metrics.shape[1]
  metrics_abs = np.abs(metrics)
  
  min_idx     = metrics_abs.argmin(axis=0)
  max_idx     = metrics_abs.argmax(axis=0)
  min_score   = metrics[min_idx, range(nb_metrics)]
  max_score   = metrics[max_idx, range(nb_metrics)]
  avg_score   = metrics_abs.mean(axis=0)
  stdev       = metrics_abs.std(axis=0)
  min_id      = [id_list[idx] for idx in list(min_idx) ]
  max_id      = [id_list[idx] for idx in list(max_idx) ]
  
  print('-'*100, file=out_file)
  
  print('Gross statistics for {} datasets'.format(nb_dsets), file=out_file)
  headings = ['Metric', 'Avg.', 'Std.', 'Min', 'Min ID', 'Max', 'Max ID']
  headings_format = '{:<10s}'*len(headings)
  print(headings_format.format(*headings), file=out_file)
  print('-'*100, file=out_file)
  
  for idx, metric_name in enumerate(metric_name_format_dict):
    metric_format = '{:<10s}{:<10.2f}{:<10.2f}{:<10.2f}{:<10s}{:<10.2f}{:<10s}'
    print(metric_format.format(metric_name, avg_score[idx], stdev[idx],
                               min_score[idx], min_id[idx],
                               max_score[idx], max_id[idx]), file=out_file)

def oc_print_sorted(id_list, anatomy_name, metric_name_format_dict={}, 
                    metrics=None, sort_by_col=0, out_file=None):
  """
  Function to sort the metrics by given column [particular metric] and print
  metrics for all datasets.
  If out_file = None, prints to Stdout otherwie to specified file.
  Metrics is numpy array with shape = nb_dsets x nb_metrics
  """
  
  nb_dsets   = metrics.shape[0]
  nb_metrics = metrics.shape[1]
  if nb_dsets < len(id_list):
    id_list = id_list[:nb_dsets]
  if sort_by_col >= nb_metrics:
    sort_by_col = 0
  
  sort_indexes = np.argsort(metrics[:,sort_by_col])
  sorted_metrics = metrics[sort_indexes]
  sorted_ids = list(np.asarray(id_list)[sort_indexes])
  
  oc_print_heading(metric_name_format_dict, out_file)
  for idx, test_id in enumerate(sorted_ids):
    oc_print_dset_score(test_id, anatomy_name, metric_name_format_dict,
                        sorted_metrics[idx], out_file)
    
    
# ----------------- Functions for printing one class metrics -----------------
# ----- Functions for printing only one metric -------------------------------
def oc_print_heading_old(out_file=None):
  """
  Function to print top level heading for one class metrics file.
  If out_file = None, print to Stdout else print to file.
  """
  print('{:<15s}{:<20s}{:<20s}'.format('Dataset','Anatomy', 'DICE Score'),
        file=out_file)
  print('-'*50, file=out_file)
  
def oc_print_dset_score_old(test_id, anatomy_name, dice_score, out_file=None):
  """
  Function to print DICE score for a particular dataset.
  If out_file = None, prints to Stdout otherwise to specified file
  """
  print('{:<15s}{:<20s}{:<2.2f}{:<0s}'.format(
      test_id, anatomy_name, float(dice_score)*100,'%'),
    file=out_file)

def oc_calc_print_aggregate_old(id_list, dice_scores, out_file=None):
  """
  Function to calculate and print aggregate scores for one class metrics.
  If out_file = None, prints to Stdout otherwie to specified file.
  """
  # Calculate and write aggregate statistics
  dice_scores = dice_scores * 100.
  min_score   = dice_scores.min()
  max_score   = dice_scores.max()
  avg_score   = dice_scores.mean()
  std         = dice_scores.std()
  min_idx     = dice_scores.argmin()
  max_idx     = dice_scores.argmax()
  min_id      = id_list[min_idx]
  max_id      = id_list[max_idx]
  
  nb_dsets = dice_scores.shape[0]
  print('-'*50, file=out_file)
  
  print('Gross statistics for {} datasets'.format(nb_dsets), file=out_file)
  print('{:<20s}{:<20s}'.format('Average', 'Std Deviation'), file=out_file)
  print('-'*50, file=out_file)
  print('{:<2.2f}{:<15s}{:<2.2f}{:<0s}'.format(avg_score,'%',std,'%'),
        file=out_file)
  
  print('-'*50, file=out_file)
  print('{:<20s}'.format('Minimum'), file=out_file)
  print('-'*50, file=out_file)
  print('{:<20s}{:<2.2f}{:<0s}'.format(min_id,min_score,'%'), file=out_file)
  
  print('-'*50, file=out_file)
  print('{:<20s}'.format('Maximum'), file=out_file)
  print('-'*50, file=out_file)
  print('{:<20s}{:<2.2f}{:<0s}'.format(max_id,max_score,'%'), file=out_file)

# ----------------- Functions for printing multi class metrics ---------------
def mc_print_dset_metrics(test_id, dice_score_list, 
                       label_dict, classes_present, out_file=None):
  """
  Print one dataset's dice scores for all present classes.
  If out_file = None, prints to stdout.
  """
  
  # --------------- Print Heading ------------------------------
  print('{:<20s}{:<20s}'.format(
    'Dataset',test_id), file=out_file)
  print('-'*50, file=out_file)
  print('{:<20s}{:<20s}'.format(
    'Anatomy','DICE Score'), file=out_file)
  
  # --------------- Print Scores for each anatomy present --------------------
  for cls_label in classes_present:
    anatomy = label_dict[np.int(cls_label)]
    dice_score = np.float(dice_score_list[np.int(cls_label)])*100
    print('{:<20s}{:<2.2f}{:<0s}'.format(anatomy, dice_score,'%'),
        file=out_file)
  print('-'*50, file=out_file)
 
def mc_calc_print_aggregate_metrics(ids_list, dice_scores, label_dict,
                               classes_present, out_file=None):
  """
  Print aggregate statistics for whole batch for each anatomy
  present in segmentation masks.
  If out_file = None, prints to Stdout else to file.
  """
  nb_dsets = dice_scores.shape[0]
  dice_scores = dice_scores * 100.
  max_scores = dice_scores.max(axis=0)
  min_scores = dice_scores.min(axis=0)
  avg_scores = dice_scores.mean(axis=0)
  std_scores = dice_scores.std(axis=0)
  min_idxs   = dice_scores.argmin(axis=0)
  max_idxs   = dice_scores.argmax(axis=0)
  min_ids    = np.asarray(ids_list)[min_idxs]
  max_ids    = np.asarray(ids_list)[max_idxs]
  
  # Print aggregate statistics for each anatomy present ----------------------
  print('Gross statistics for {} datasets'.format(nb_dsets), file=out_file)
  print('-'*50, file=out_file)
  print('{:<20s}{:<15s}{:<15s}{:<15s}{:<15s}{:<15s}{:<15s}'.format(
      'Anatomy', 'Average', 'Std', 'Min', 'ID', 'Max', 'ID'), file=out_file)
  
  for idx, cls_label in enumerate(classes_present):
    anatomy = label_dict[np.int(cls_label)]
    print('{:<20s}{:<2.2f}{:<11s}{:<2.2f}{:<11s}{:<2.2f}{:<11s}{:<15s}'\
          '{:<2.2f}{:<11s}{:<15s}'.format(
        anatomy, avg_scores[idx], '%', std_scores[idx], '%',
        min_scores[idx], '%', min_ids[idx], 
        max_scores[idx], '%', max_ids[idx]), file=out_file)
  print('-'*50, file=out_file)
  
# ---------- Functions for printing prediction overlap metrics ---------------
  
def print_confusion_matrix(heading, confusion_matrix, label_dict,
                           out_file=None):
  # Print heading for current dataset
  num_classes = len(label_dict) - 1
  print(heading, file=out_file)
  print('-'*100, file=out_file)
  
  # Print top row for current dset confusion matrix
  print('{:<20s}{:<20s}{:<20s}{:<20s}{:<20s}'.format(
      '\\', label_dict[1], label_dict[2], label_dict[3], label_dict[4]),
  file=out_file)
  
  # Print confusion matrix now
  for i in range(num_classes):
    row = '{:<20s}'.format(label_dict[i+1])
    for j in range(num_classes):
      col = '{:<20d}'.format(confusion_matrix[i,j])
      row += col
    # Print row
    print(row, file=out_file)
  # End printing confusion matrix
  print('='*100, file=out_file)
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  