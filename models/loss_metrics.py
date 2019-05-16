#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:52:52 2019
Implementation of loss metrics that can be tracked while training models.

@author: ndahiya
"""

from keras import backend as K
from os import path
import sys
sys.path.insert(0, path.abspath('.'))
from models.loss_functions import soft_sorensen_dice

def dice_1(y_true, y_pred):
  """
  Dice loss for class 1
  """
  batch_dice_loss = soft_sorensen_dice(y_true, y_pred, axis=[2,3])
  mean_dice_loss  = K.mean(batch_dice_loss, axis=0)
  return mean_dice_loss[1]

def dice_2(y_true, y_pred):
  """
  Dice loss for class 2
  """
  batch_dice_loss = soft_sorensen_dice(y_true, y_pred, axis=[2,3])
  mean_dice_loss  = K.mean(batch_dice_loss, axis=0)
  return mean_dice_loss[2]

def dice_3(y_true, y_pred):
  """
  Dice loss for class 3
  """
  batch_dice_loss = soft_sorensen_dice(y_true, y_pred, axis=[2,3])
  mean_dice_loss  = K.mean(batch_dice_loss, axis=0)
  return mean_dice_loss[3]

def dice_4(y_true, y_pred):
  """
  Dice loss for class 4
  """
  batch_dice_loss = soft_sorensen_dice(y_true, y_pred, axis=[2,3])
  mean_dice_loss  = K.mean(batch_dice_loss, axis=0)
  return mean_dice_loss[4]

def dice_coef(y_true, y_pred):
  """
  Calculate DICE coefficient score for predicted mask. Expects true and
  predicted masks to be binary arrays [0/1]. 3D Volumetric DICE for one class.
  Take mean over batch
  
  # Arguments:
      y_true: numpy array of true targets, y_true.shape = [slices, H, W]
      y_pred: numpy array of predicted targets, y_pred.shape = [slices, H, W]
  # Returns:
      Scalar DICE coefficient for each label in the range [0 1]
  """
  batch_dice_loss = soft_sorensen_dice(y_true, y_pred, axis=[1,2,3,4])
  mean_dice_loss  = K.mean(batch_dice_loss, axis=0)
  return mean_dice_loss