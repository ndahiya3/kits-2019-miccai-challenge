#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:01:20 2019
Unet model using dilated convolutions in the downsampling blocks and bottleneck
layer. Using following reference:
  https://blog.insightdatascience.com/heart-disease-diagnosis-with-deep-learning-c2d92c27e730
  
Smith and Nephew data has higher resoluton in 2D. Consequently, we can use more
downsampling blocks to capture higher resolution features.

Also refactoring the Unet implementation to make it more flexible and readable.
@author: ndahiya
"""
from keras.models import Model
from keras.layers import Conv2D, Dropout, MaxPool2D, UpSampling2D, concatenate
from keras.layers import Input, Permute, Activation
from keras.optimizers import Adam
from models.loss_functions import dice_coef_loss
from models.loss_functions import dice_coef
from models.loss_functions import mc_sorensen_dice_loss, mc_tversky_loss
from models.loss_metrics import dice_1, dice_2, dice_3, dice_4

def conv_block(in_tensor, num_features, dilated_conv=True, batch_norm=False):
  # Convolution block at a particular Unet "level"
  dilation_rate = 1
  encoder = Conv2D(num_features, 5, padding='same',
                   data_format='channels_first', activation='relu',
                   kernel_initializer='VarianceScaling',
                   dilation_rate=dilation_rate)(in_tensor)
  encoder = Dropout(rate=0.1)(encoder)
  if dilated_conv:
    dilation_rate = 2
  encoder = Conv2D(num_features, 5, padding='same',
                   data_format='channels_first', activation='relu',
                   kernel_initializer='VarianceScaling',
                   dilation_rate=dilation_rate)(encoder)
  return encoder

def encoder_block(in_tensor, num_features, dilated_conv=True, batch_norm=False):
  # Full Unet encoder block
  # Conv  --> Dropout --> (Dilated) Conv --> Maxpool
  encoder = conv_block(in_tensor, num_features, dilated_conv, batch_norm)
  encoder_pool = MaxPool2D(pool_size=(2,2), padding='valid',
                           data_format='channels_first')(encoder)
  return encoder_pool, encoder

def decoder_block(in_tensor, concat_tensor, num_filters):
  # Expand and Concatenate part of Network
  # Upsample --> Merge | Conv  --> Dropout --> Conv
  decoder = UpSampling2D(size=(2,2),data_format='channels_first')(in_tensor)
  decoder = concatenate([decoder,concat_tensor],axis=1)

  decoder = Conv2D(num_filters, 5, padding='same', 
                   data_format='channels_first', activation='relu',
                   kernel_initializer='VarianceScaling')(decoder)
  decoder = Dropout(rate=0.1)(decoder)
  decoder = Conv2D(num_filters, 5, padding='same', 
                   data_format='channels_first', activation='relu',
                   kernel_initializer='VarianceScaling')(decoder)
  return decoder

def unet_model_deeper_dilated_conv(pretrained_weights=None,
                                   input_size=(1,512,512), num_classes=1):
  print('Deeper Dilated Convolution:')
  input_img = Input(shape=input_size)                           # 512
  encoder_pool0, encoder0 = encoder_block(input_img, 32)        # 256
  encoder_pool1, encoder1 = encoder_block(encoder_pool0, 64)    # 128
  encoder_pool2, encoder2 = encoder_block(encoder_pool1, 128)   # 64
  encoder_pool3, encoder3 = encoder_block(encoder_pool2, 256)   # 32
  encoder_pool4, encoder4 = encoder_block(encoder_pool3, 512)   # 16
  
  center = conv_block(encoder_pool4, 1024)
  
  decoder4 = decoder_block(center,   encoder4, 512)
  decoder3 = decoder_block(decoder4, encoder3, 256)
  decoder2 = decoder_block(decoder3, encoder2, 128)
  decoder1 = decoder_block(decoder2, encoder1, 64)
  decoder0 = decoder_block(decoder1, encoder0, 32)
  
  # Output
  output = Conv2D(num_classes, 1, padding='same', data_format='channels_first',
                      kernel_initializer='VarianceScaling')(decoder0)
  
  if num_classes > 1:
    activation_type = 'softmax'
    output = Permute((2,3,1))(output) # Classes last
    output = Activation(activation_type)(output)
    output = Permute((3,1,2))(output) # Classes first again
  else:
    activation_type = 'sigmoid'
    output = Activation(activation_type)(output)
    
  model = Model(inputs=input_img, outputs=output)
  
  if num_classes > 1:
    track_metrics = [dice_1, dice_2]#, dice_3, dice_4]
    #model.compile(optimizer=Adam(lr=1e-4), loss=my_categorical_crossentropy, metrics=track_metrics)
    #model.compile(optimizer=Adam(lr=1e-6), loss=mc_generalized_dice_loss, metrics=track_metrics)
    model.compile(optimizer=Adam(lr=1e-5), loss=mc_tversky_loss, metrics=track_metrics)
    #model.compile(optimizer=Adam(lr=1e-4), loss=mc_sorensen_dice_loss, metrics=track_metrics)
    #model.compile(optimizer=Adam(lr=1e-4), loss=mc_weighted_categorical_crossentropy, metrics=track_metrics)
    #model.compile(optimizer=Adam(lr=1e-5), loss=mc_focal_tversky, metrics=track_metrics)
    #model.compile(optimizer=Adam(lr=1e-4), loss=mc_focal_loss, metrics=track_metrics)
  else:
    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=['accuracy', dice_coef])
    #model.compile(optimizer=Adam(lr=1e-4), loss = 'binary_crossentropy', metrics=['accuracy'])

  if (pretrained_weights):
      model.load_weights(pretrained_weights)

  return model











































