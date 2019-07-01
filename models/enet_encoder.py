#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:54:17 2019
Encoder for the ENet segmentation model. ENet Encoder consists of an initial
block followed by different configurations of Bottleneck layers including
downsampling, dilated convolutions, asymmetric convolutions etc.
@author: ndahiya
"""

from keras.layers import Conv2D, MaxPooling2D, concatenate, BatchNormalization
from keras.layers import PReLU, SpatialDropout2D, Permute, ZeroPadding2D
from keras.layers import add

def initial_block(in_tensor, kernel_size=(3,3), strides=(2,2), nb_filters=13):
  
  conv = Conv2D(nb_filters,kernel_size=kernel_size, strides=strides, 
                padding='same', kernel_initializer='VarianceScaling')(in_tensor)
  max_pool = MaxPooling2D(pool_size=(2,2), padding='valid')(in_tensor)
  merged   = concatenate([conv, max_pool], axis=3)
  
  return merged

def bottleneck_layer(in_tensor, out_filters, dilation=1, asymmetric=False, 
                     downsample=False, dropout_rate=0.1, projection_ratio=4):
  
  reduced_depth = int(out_filters // projection_ratio)
  
  # If downsampling the first [1x1] convolution in the right branch is replaced 
  # by [2,2] convolution with stride [2x2]
  if downsample is True:
    kernel_size = (2,2)
    strides = (2,2)
  else:
    kernel_size = (1,1)
    strides = (1,1)
    
  encoder = Conv2D(reduced_depth, kernel_size=kernel_size, strides=strides, 
                   kernel_initializer='VarianceScaling', use_bias=False)(in_tensor)
  encoder = BatchNormalization(momentum=0.1)(encoder)
  encoder = PReLU(shared_axes=[1,2])(encoder)
  
  # 2nd Regular Convolution
  if asymmetric is True:
    encoder = Conv2D(reduced_depth, kernel_size=(1,5), padding='same', 
                     use_bias=False)(encoder)
    encoder = Conv2D(reduced_depth, kernel_size=(5,1), padding='same', 
                     use_bias=True)(encoder)
  else:
    encoder = Conv2D(reduced_depth, kernel_size=(3,3), dilation_rate=dilation,
                     padding='same', kernel_initializer='VarianceScaling')(encoder)
      
  encoder = BatchNormalization(momentum=0.1)(encoder)
  encoder = PReLU(shared_axes=[1,2])(encoder)
  
  # Third [1x1] convolution
  encoder = Conv2D(out_filters, (1,1), use_bias=False,
                   kernel_initializer='VarianceScaling')(encoder)
  encoder = BatchNormalization(momentum=0.1)(encoder)
  
  encoder = SpatialDropout2D(rate=dropout_rate)(encoder)
  
  # Left branch
  # If downsample we max pool first and then add padding in channels dim to make
  # number of channels/features same before adding the two branches
  left_encoder = in_tensor
  if downsample is True:
    left_encoder = MaxPooling2D(pool_size=(2,2))(left_encoder)
  
  in_channels = in_tensor.get_shape().as_list()[3]
  
  # Zero pad by moving channels dim to width dim
  if in_channels != out_filters:
    padding = out_filters - in_channels
    left_encoder = Permute(dims=(1,3,2))(left_encoder)
    left_encoder = ZeroPadding2D(padding=((0,0),(0, padding)))(left_encoder)
    left_encoder = Permute(dims=(1,3,2))(left_encoder)
  
  encoder = add([left_encoder, encoder])
  encoder = PReLU(shared_axes=[1,2])(encoder)
  
  return encoder
  
def build_encoder(in_tensor, dropout_rate=0.01):
  # Build the encoder with Initial Block and Three stages each comprised of
  # bottleneck layers of different types
  
  # Initial block
  encoder = initial_block(in_tensor, nb_filters=15)
  encoder = BatchNormalization(momentum=0.1)(encoder)
  encoder = PReLU(shared_axes=[1,2])(encoder)
  
  # Stage 1
  encoder = bottleneck_layer(encoder, out_filters=64, downsample=True, 
                             dropout_rate=dropout_rate)
  for i in range(4): # bottleneck 1.x
    encoder = bottleneck_layer(encoder, out_filters=64, dropout_rate=dropout_rate)
    
  # Stage 2 and Stage 3, bottleneck 2.x and 3.x
  # Default dropout_rate = 0.1 from now onwards
  encoder = bottleneck_layer(encoder, out_filters=128, downsample=True) #bottleneck 2.0
  
  for i in range(2):
    encoder = bottleneck_layer(encoder, out_filters=128)                  #bottleneck 2.1
    encoder = bottleneck_layer(encoder, out_filters=128, dilation=2)      #bottleneck 2.2
    encoder = bottleneck_layer(encoder, out_filters=128, asymmetric=True) #bottleneck 2.3
    encoder = bottleneck_layer(encoder, out_filters=128, dilation=4)      #bottleneck 2.4
    encoder = bottleneck_layer(encoder, out_filters=128)                  #bottleneck 2.5
    encoder = bottleneck_layer(encoder, out_filters=128, dilation=8)      #bottleneck 2.6
    encoder = bottleneck_layer(encoder, out_filters=128, asymmetric=True) #bottleneck 2.7
    encoder = bottleneck_layer(encoder, out_filters=128, dilation=16)     #bottleneck 2.8
  
  return encoder
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  