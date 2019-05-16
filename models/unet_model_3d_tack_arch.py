#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 12:59:20 2019
3D Unet model architecure Tack et. al. architecture.
Ref: https://www.sciencedirect.com/science/article/pii/S1361841518304882
@author: ndahiya
"""

from keras.models import Model
from keras.layers import Input, Conv3D, MaxPool3D, UpSampling3D, concatenate
from keras.layers import Dropout
from keras.optimizers import Adam
from models.loss_functions import dice_coef_loss
from models.loss_metrics import dice_coef
from models.loss_functions import tversky_loss_3d

def conv_block(in_tensor, num_features, kernel_size=(5,5,5), batch_norm=False):
  # Convolution block at a particular Unet "level"
  encoder = Conv3D(num_features, kernel_size, padding='same',
                   data_format='channels_first', activation='relu',
                   kernel_initializer='VarianceScaling')(in_tensor)
  encoder = Dropout(rate=0.1)(encoder)
  encoder = Conv3D(num_features, kernel_size, padding='same',
                   data_format='channels_first', activation='relu',
                   kernel_initializer='VarianceScaling')(encoder)
  return encoder

def encoder_block(in_tensor, num_features, kernel_size=(5,5,5),
                  max_pool_size=(2,2,2), batch_norm=False):
  # Full Unet encoder block
  # Conv  --> Dropout --> Conv --> Maxpool
  encoder = conv_block(in_tensor, num_features, kernel_size, batch_norm)
  encoder_pool = MaxPool3D(pool_size=max_pool_size, padding='valid',
                           data_format='channels_first')(encoder)
  return encoder_pool, encoder

def decoder_block(in_tensor, concat_tensor, num_filters, conv_kernel_size,
                  up_sample_size):
  # Expand and Concatenate part of Network
  # Upsample --> Merge | Conv  --> Dropout --> Conv
  decoder = UpSampling3D(size=up_sample_size,data_format='channels_first')(in_tensor)
  decoder = concatenate([decoder,concat_tensor],axis=1)

  decoder = Conv3D(num_filters, kernel_size=conv_kernel_size, padding='same', 
                   data_format='channels_first', activation='relu',
                   kernel_initializer='VarianceScaling')(decoder)
  decoder = Dropout(rate=0.1)(decoder)
  decoder = Conv3D(num_filters, kernel_size=conv_kernel_size, padding='same', 
                   data_format='channels_first', activation='relu',
                   kernel_initializer='VarianceScaling')(decoder)
  return decoder

def unet_model_3d_tack_arch(pretrained_weights=None, input_size = (1,16,64,64)):
  
  input_img = Input(shape=input_size)                                             # 16
  encoder_pool0, encoder0 = encoder_block(input_img, 32, kernel_size=(5,5,5),
                                          max_pool_size=(2,2,2))                  # 8
  encoder_pool1, encoder1 = encoder_block(encoder_pool0, 64, kernel_size=(5,5,5),
                                          max_pool_size=(2,2,2))                  # 4
  encoder_pool2, encoder2 = encoder_block(encoder_pool1, 128, kernel_size=(3,5,5),
                                          max_pool_size=(1,2,2))                  # 4
  center = conv_block(encoder_pool2, 256, kernel_size=(3,5,5))                    # 4
  
  decoder2 = decoder_block(center,   encoder2, 128, conv_kernel_size=(3,5,5),
                           up_sample_size=(1,2,2))
  decoder1 = decoder_block(decoder2, encoder1, 64, conv_kernel_size=(5,5,5),
                           up_sample_size=(2,2,2))
  decoder0 = decoder_block(decoder1, encoder0, 32, conv_kernel_size=(5,5,5),
                           up_sample_size=(2,2,2))
  
  # Output
  output = Conv3D(1, kernel_size=(1,1,1), padding='same',
                  data_format='channels_first', 
                  kernel_initializer='VarianceScaling',
                  activation='sigmoid')(decoder0)
  
  model = Model(inputs=input_img, outputs=output)
  model.compile(optimizer=Adam(lr=1e-5), loss=tversky_loss_3d, metrics=[dice_coef])
  
  if (pretrained_weights):
    model.load_weights(pretrained_weights)

  return model
  
