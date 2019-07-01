#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:54:39 2019
Decoder for ENet. It is composed of two types of bottleneck layers, regular and
upsampling.
@author: ndahiya
"""
from keras.layers import Conv2D, add, BatchNormalization, UpSampling2D
from keras.layers import Activation, Conv2DTranspose

def bottleneck_layer(in_tensor, out_filters, upsample=False, 
                              dropout_rate=0.1, projection_ratio=4):
  
  reduced_depth = int(out_filters // projection_ratio)
  
  bottleneck = Conv2D(reduced_depth, kernel_size=(1,1),
                      kernel_initializer='VarianceScaling', use_bias=False)(in_tensor)
  bottleneck = BatchNormalization(momentum=0.1)(bottleneck)
  bottleneck = Activation('relu')(bottleneck)
  
  if upsample is True:
    bottleneck = Conv2DTranspose(filters=reduced_depth, kernel_size=(3, 3), 
                                 strides=(2, 2), padding='same', 
                                 kernel_initializer='VarianceScaling' )(bottleneck)
  else:
    bottleneck = Conv2D(reduced_depth, kernel_size=(3,3), 
                   padding='same', kernel_initializer='VarianceScaling')(bottleneck)
      
  bottleneck = BatchNormalization(momentum=0.1)(bottleneck)
  bottleneck = Activation('relu')(bottleneck)
  
  bottleneck = Conv2D(out_filters, kernel_size=(1, 1), padding='same', 
                      use_bias=False)(bottleneck)
  bottleneck = BatchNormalization(momentum=0.1)(bottleneck)
  
  left_bottleneck = in_tensor
  if upsample is True:
    left_bottleneck = Conv2D(out_filters, kernel_size=(1, 1), padding='same', 
                           use_bias=False)(left_bottleneck)
    left_bottleneck = BatchNormalization(momentum=0.1)(left_bottleneck)
    left_bottleneck = UpSampling2D(size=(2,2))(left_bottleneck)
  
  bottleneck = add([bottleneck, left_bottleneck])
  bottleneck = Activation('relu')(bottleneck)
  
  return bottleneck
  
def build_decoder(encoder, num_classes):
  
  decoder = bottleneck_layer(encoder, out_filters=64, upsample=True)  #bottleneck 4.0
  decoder = bottleneck_layer(decoder, out_filters=64)                 #bottleneck 4.1 
  decoder = bottleneck_layer(decoder, out_filters=64)                 #bottleneck 4.2
  
  decoder = bottleneck_layer(decoder, out_filters=16, upsample=True)  #bottleneck 5.0
  decoder = bottleneck_layer(decoder, out_filters=16)                 #bottleneck 5.1
  
  decoder = Conv2DTranspose(num_classes, kernel_size=(2,2), strides=(2,2), 
                            padding='same')(decoder)
  
  return decoder
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  