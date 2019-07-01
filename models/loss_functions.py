#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:52:05 2019
Implementation of different loss functions to be minimized for training
CNNs for knee segmentations. Basic loss functions such as cross-entropy,
mse, etc are already implemented in Keras etc.
DICE and Jaccard index are typically used in medical image processing.
@author: ndahiya
"""

from keras import backend as K
import tensorflow as tf
import numpy as np

def dice_coef(y_true, y_pred, smooth=1.):
  """
  DICE coefficient: Given 2 sets X and Y
            2* |X ^ Y|
    DSC = -----------------
            |X| + |Y|
  where |*| = cardinality and ^ == intersection

  Also interpreted as:
                2*TP
    DSC = -------------------
            2*TP + FP + FN
  Smooth argument makes it differentiable and avoid division by 0

  # Arguments:
      y_true: tensor of true targets
      y_pred: tensor of predicted targets
  # Returns:
      Tensor with one scalar loss averaged over batch dimension
  """

  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
  
  return score

def dice_coef_loss(y_true, y_pred):
  """
  Binary DICE coefficient based loss.

  # Arguments:
      y_true: tensor of true targets
      y_pred: tensor of predicted targets
  # Returns:
      Tensor with one scalar loss averaged over batch dimension with output
      range [0 1]
  """
  loss = 1 - dice_coef(y_true, y_pred)
  return loss

def my_categorical_crossentropy(y_true, y_pred):
  # Categorical cross entropy takes care of channels_first image ordering
  if K.image_data_format() == 'channels_first':
    channels_axis = 1
  else:
    channels_axis = 3
  loss = K.categorical_crossentropy(y_true, y_pred, from_logits=False, axis=channels_axis)
  return loss

def tversky_index_3d(y_true, y_pred):
  """
  Calculate Tversky index for 3D data [one class or multiclass]
  """
  alpha = 0.3
  beta  = 0.7
  #alpha = 0.5
  #beta  = 0.5
  # Input tensors have shape [batch_size, classes, depth, height, width]
  y_true_ch_last = K.permute_dimensions(y_true, pattern=(0,2,3,4,1))
  y_pred_ch_last = K.permute_dimensions(y_pred, pattern=(0,2,3,4,1))
  
  ones = K.ones_like(y_true_ch_last)
  p0 = y_pred_ch_last      # proba that voxels are class i
  p1 = ones-y_pred_ch_last # proba that voxels are not class i
  g0 = y_true_ch_last
  g1 = ones-y_true_ch_last
  
  num = K.sum(p0*g0, axis=[0,1,2,3])
  den = num + alpha*K.sum(p0*g1,axis=[0,1,2,3]) + beta*K.sum(p1*g0,axis=[0,1,2,3])
  
  tversky_index = num/den
  return tversky_index

def tversky_loss_3d(y_true, y_pred):
  """
  Calculate Tversky loss based on tversky index. alpha,beta tunable parameters
  """
  tversky_index = tversky_index_3d(y_true, y_pred)
  
  T = K.sum(tversky_index) # when summing over classes, T has dynamic range [0 Ncl]
    
  Ncl = K.cast(K.int_shape(y_pred)[1], 'float32')
  return Ncl-T

def soft_sorensen_dice(y_true, y_pred, axis=None, smooth=1):
  """
  Sorensen Dice score. Soft because uses softmax probabilities in y_pred
  instead of converting to hard binary prediction masks.
  Ref: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
  """
  intersection = K.sum(y_true * y_pred, axis=axis)
  area_true = K.sum(y_true, axis=axis)
  area_pred = K.sum(y_pred, axis=axis)
  
  return (2 * intersection + smooth) / (area_true + area_pred + smooth)

def soft_jaccard_index(y_true, y_pred, axis=None, smooth=1):
  """
  Jaccard similarity index. Soft because uses softmax probabilities in y_pred
  instead of converting to hard binary prediction masks. Closely related to
  sorensen dice.
  Ref: https://en.wikipedia.org/wiki/Jaccard_index
  """
  intersection = K.sum(y_true * y_pred, axis=axis)
  area_true = K.sum(y_true, axis=axis)
  area_pred = K.sum(y_pred, axis=axis)
  union = area_true + area_pred - intersection
  
  return (intersection + smooth) / ( union + smooth)

def mc_sorensen_dice_loss(y_true, y_pred):
  """
  Sorenson class weighted DICE similarity loss for multiclass segmentation.
  Class weights are manually set. Also called dense-DICE loss?
  """  
  # Input tensors have shape [batch_size, classes, height, width]
  # User must input list of weights with length equal to number of classes
  
  batch_dice_coefs = soft_sorensen_dice(y_true, y_pred, axis=[2, 3])
  dice_coefs = K.mean(batch_dice_coefs, axis=0) # Mean over batch axis
  
  weights = [0, 1, 1]#, 1, 3]
  w = K.constant(weights)/sum(weights)
  print(K.eval(w))
  return 1 - K.sum(w * dice_coefs)

def mc_jacaard_loss(y_true, y_pred):
  """
  Jaccard similarity index bbased loss for multiclass segmentation.
  Class weights are manually set.
  """  
  batch_jaccard_coefs = soft_jaccard_index(y_true, y_pred, axis=[2, 3])
  jaccard_coefs = K.mean(batch_jaccard_coefs, axis=0) # Mean over batch axis

  w = K.constant([0, 1, 2, 1, 3])/7

  return 1 - K.sum(w * jaccard_coefs)

def mc_tversky_index(y_true, y_pred):
  """
  Multiclass Tversky loss. Weighs the FN more than FP and helps segment
  smaller objects.
  alpha = beta = 0.5 reduces to DICE over whole batch.
  Ref: https://arxiv.org/abs/1706.05721
  """
  alpha = 0.3
  beta  = 0.7
  #alpha = 0.5
  #beta  = 0.5
  if K.image_data_format() == 'channels_first':
    # Input tensors have shape [batch_size, classes, height, width]
    y_true_ch_last = K.permute_dimensions(y_true, pattern=(0,2,3,1))
    y_pred_ch_last = K.permute_dimensions(y_pred, pattern=(0,2,3,1))
  else:
    y_true_ch_last = y_true
    y_pred_ch_last = y_pred
  
  ones = K.ones_like(y_true_ch_last)
  p0 = y_pred_ch_last      # proba that voxels are class i
  p1 = ones-y_pred_ch_last # proba that voxels are not class i
  g0 = y_true_ch_last
  g1 = ones-y_true_ch_last
  
  num = K.sum(p0*g0, axis=[0,1,2])
  den = num + alpha*K.sum(p0*g1,axis=[0,1,2]) + beta*K.sum(p1*g0,axis=[0,1,2])
  
  tversky_index = num/den
  return tversky_index

def mc_tversky_loss(y_true, y_pred):
  
  tversky_index = mc_tversky_index(y_true, y_pred)
  
  T = K.sum(tversky_index) # when summing over classes, T has dynamic range [0 Ncl]
  
  if K.image_data_format() == 'channels_first':
    Ncl = K.cast(K.int_shape(y_pred)[1], 'float32')
  else:
    Ncl = K.cast(K.int_shape(y_pred)[3], 'float32')
  return Ncl-T

def mc_focal_tversky(y_true, y_pred):
  """
  Focal + Tversky loss
  Ref: https://arxiv.org/pdf/1810.07842.pdf
  """
  gamma = 4.0/3.0
  T = mc_tversky_index(y_true, y_pred)
  ones = K.ones_like(T)
  one_minus_tversky = ones - T
  focal_tversky = K.pow(one_minus_tversky, tf.reciprocal(gamma))
  focal_tversky_loss = K.sum(focal_tversky)
  
  return focal_tversky_loss

def mc_weighted_categorical_crossentropy(y_true, y_pred):
  """
  Weighted multiclass/categorical cross entropy loss, averaged over batch.
  binary crossentropy = -[y(log(p)) + (1-y)log(1-p)]
  mc crossentropy = -Sum_over_classes(y_true_class*log(y_pred_class))
  Expects y_pred to sum to 1 over classes i.e. output of softmax
  """
  # Input tensors have shape [batch_size, classes, height, width]
  y_true_ch_last = K.permute_dimensions(y_true, pattern=(0,2,3,1))
  y_pred_ch_last = K.permute_dimensions(y_pred, pattern=(0,2,3,1))
  
  ndim = K.ndim(y_pred_ch_last)
  nclasses = K.int_shape(y_pred_ch_last)[-1]
  eps = 1e-8
  y_pred_ch_last = K.clip(y_pred_ch_last, eps, 1-eps)
  
  #weights = [0.5, 1.0, 2.0, 1.0, 2.0]
  weights = [0.5, 1.0, 1.0]#, 1.0, 2.0]
  w = K.constant(weights)*(nclasses/sum(weights))
  
  # Average over batch + spatial except classes axis
  cross_entropies = -K.mean(y_true_ch_last * K.log(y_pred_ch_last),
                            axis=tuple(range(ndim-1)))
  
  return K.sum(w * cross_entropies)

def mc_focal_loss(y_true, y_pred):
  """
  Focal loss based on categorical cross entropy.
  Ref: https://arxiv.org/abs/1708.02002
  """
  y_true_ch_last = K.permute_dimensions(y_true, pattern=(0,2,3,1))
  y_pred_ch_last = K.permute_dimensions(y_pred, pattern=(0,2,3,1))
  
  ndim = K.ndim(y_pred_ch_last)
  nclasses = K.int_shape(y_pred_ch_last)[-1]
  eps = 1e-8
  y_pred_ch_last = K.clip(y_pred_ch_last, eps, 1-eps)
  
  weights = [0.3, 1.0, 1.0, 1.0, 2.0]
  w = K.constant(weights)*(nclasses/sum(weights))
  
  gamma = 2.0 # Gamma = 0 gives simple categorical cross entropy loss
  ones = tf.ones_like(y_pred_ch_last)
  one_minus_pred = ones - y_pred_ch_last
  
  # Average over batch + spatial except classes axis
  focal_part = K.pow(one_minus_pred, gamma)
  cross_entropies = -K.mean(focal_part * y_true_ch_last * K.log(y_pred_ch_last),
                            axis=tuple(range(ndim-1)))
  
  return K.sum(w * cross_entropies)
  
def mc_generalized_dice_loss(y_true, y_pred):
  """
  Generalized DICE loss. Weighted by inverse of true class vol^2
  Ref:https://arxiv.org/abs/1707.03237
  """
  # Input tensors have shape [batch_size, classes, height, width]
  y_true_ch_last = K.permute_dimensions(y_true, pattern=(0,2,3,1))
  y_pred_ch_last = K.permute_dimensions(y_pred, pattern=(0,2,3,1))
  
  ref_vol_batch = K.sum(y_true_ch_last, axis=[1,2])
  ref_vol_batch_sq = K.square(ref_vol_batch)
  weights_batch = tf.reciprocal(ref_vol_batch_sq)
  new_weights_batch = tf.where(tf.is_inf(weights_batch), tf.zeros_like(weights_batch),
                               weights_batch)
  weights_batch = tf.where(tf.is_inf(weights_batch), tf.ones_like(weights_batch) * 
                           tf.reduce_max(new_weights_batch), weights_batch)
  intersection_batch = 2*weights_batch*K.sum(y_true_ch_last*y_pred_ch_last, axis=[1,2])
  numerator_batch = K.sum(intersection_batch, axis=[1]) # Add weighted intersections across classes
  
  pred_vol_batch = K.sum(y_pred_ch_last, axis=[1,2])
  union_batch = weights_batch * (ref_vol_batch + pred_vol_batch)
  den_batch = K.sum(union_batch, axis=[1])# Add weighted unions across classes
  
  gen_dice_score_batch = numerator_batch / den_batch
  #gen_dice_score_batch = tf.where(tf.is_nan)
  GDL = 1 - K.mean(gen_dice_score_batch)
  
  return GDL