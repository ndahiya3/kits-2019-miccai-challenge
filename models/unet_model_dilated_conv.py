#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:46:14 2019
Unet model using dilated convolutions in the downsampling blocks and bottleneck
layer. Using following reference:
  https://blog.insightdatascience.com/heart-disease-diagnosis-with-deep-learning-c2d92c27e730
@author: ndahiya
"""

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, concatenate
from keras.layers import Dropout, Activation, Permute
from keras.optimizers import Adam

from models.loss_functions import dice_coef_loss
from models.loss_functions import dice_coef
from models.loss_functions import my_categorical_crossentropy
from models.loss_functions import mc_sorensen_dice_loss, mc_tversky_loss
from models.loss_functions import mc_jacaard_loss, mc_weighted_categorical_crossentropy
from models.loss_functions import mc_generalized_dice_loss, mc_focal_tversky
from models.loss_functions import mc_focal_loss
from models.loss_metrics import dice_1, dice_2, dice_3, dice_4

def unet_model_dilated_conv(pretrained_weights=None, input_size = (1,512,512),
                         num_classes=1):
  print('Dilated Convolution:')
  input_img = Input(shape=input_size)

  # Contraction network
  # Conv  --> Dropout --> (Dilated) Conv --> Maxpool

  conv2d_1 = Conv2D(32, 5, padding='same', data_format='channels_first', activation='relu',
                    kernel_initializer='VarianceScaling', dilation_rate=1)(input_img)
  dropout_1 = Dropout(rate=0.1)(conv2d_1)
  conv2d_2 = Conv2D(32, 5, padding='same', data_format='channels_first', activation='relu',
                    kernel_initializer='VarianceScaling', dilation_rate=2)(dropout_1)

  max_pooling2d_1 = MaxPool2D(pool_size=(2,2),padding='valid',data_format='channels_first')(conv2d_2)

  conv2d_3 = Conv2D(64, 5, padding='same', data_format='channels_first', activation='relu',
                    kernel_initializer='VarianceScaling', dilation_rate=1)(max_pooling2d_1)
  dropout_2 = Dropout(rate=0.1)(conv2d_3)
  conv2d_4 = Conv2D(64, 5, padding='same', data_format='channels_first', activation='relu',
                    kernel_initializer='VarianceScaling', dilation_rate=2)(dropout_2)

  max_pooling2d_2 = MaxPool2D(pool_size=(2, 2), padding='valid', data_format='channels_first')(conv2d_4)

  conv2d_5 = Conv2D(128, 5, padding='same', data_format='channels_first', activation='relu',
                    kernel_initializer='VarianceScaling', dilation_rate=1)(max_pooling2d_2)
  dropout_3 = Dropout(rate=0.1)(conv2d_5)
  conv2d_6 = Conv2D(128, 5, padding='same', data_format='channels_first', activation='relu',
                    kernel_initializer='VarianceScaling', dilation_rate=2)(dropout_3)

  max_pooling2d_3 = MaxPool2D(pool_size=(2, 2), padding='valid', data_format='channels_first')(conv2d_6)

  conv2d_7 = Conv2D(256, 5, padding='same', data_format='channels_first', activation='relu',
                    kernel_initializer='VarianceScaling', dilation_rate=1)(max_pooling2d_3)
  dropout_4 = Dropout(rate=0.1)(conv2d_7)
  conv2d_8 = Conv2D(256, 5, padding='same', data_format='channels_first', activation='relu',
                    kernel_initializer='VarianceScaling', dilation_rate=2)(dropout_4)

  # Expand and Concatenate part of Network
  # Upsample --> Merge | Conv  --> Dropout --> Conv
  up_sampling2d_1 = UpSampling2D(size=(2,2),data_format='channels_first')(conv2d_8)
  merge_1 = concatenate([up_sampling2d_1,conv2d_6],axis=1)

  conv2d_9 = Conv2D(128, 5, padding='same', data_format='channels_first', activation='relu',
                    kernel_initializer='VarianceScaling')(merge_1)
  dropout_5 = Dropout(rate=0.1)(conv2d_9)
  conv2d_10 = Conv2D(128, 5, padding='same', data_format='channels_first', activation='relu',
                    kernel_initializer='VarianceScaling')(dropout_5)

  up_sampling2d_2 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv2d_10)
  merge_2 = concatenate([up_sampling2d_2, conv2d_4], axis=1)

  conv2d_11 = Conv2D(64, 5, padding='same', data_format='channels_first', activation='relu',
                    kernel_initializer='VarianceScaling')(merge_2)
  dropout_6 = Dropout(rate=0.1)(conv2d_11)
  conv2d_12 = Conv2D(64, 5, padding='same', data_format='channels_first', activation='relu',
                     kernel_initializer='VarianceScaling')(dropout_6)

  up_sampling2d_3 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv2d_12)
  merge_3 = concatenate([up_sampling2d_3, conv2d_2], axis=1)

  conv2d_13 = Conv2D(32, 5, padding='same', data_format='channels_first', activation='relu',
                     kernel_initializer='VarianceScaling')(merge_3)
  dropout_7 = Dropout(rate=0.1)(conv2d_13)
  conv2d_14 = Conv2D(32, 5, padding='same', data_format='channels_first', activation='relu',
                     kernel_initializer='VarianceScaling')(dropout_7)

  # Output
  conv2d_15 = Conv2D(num_classes, 1, padding='same', data_format='channels_first',
                      kernel_initializer='VarianceScaling')(conv2d_14)

  if num_classes > 1:
    activation_type = 'softmax'
    conv2d_15 = Permute((2,3,1))(conv2d_15) # Classes last
    conv2d_15 = Activation(activation_type)(conv2d_15)
    conv2d_15 = Permute((3,1,2))(conv2d_15) # Classes first again
  else:
    activation_type = 'sigmoid'
    conv2d_15 = Activation(activation_type)(conv2d_15)

  model = Model(inputs=input_img, outputs=conv2d_15)

  if num_classes > 1:
    track_metrics = [dice_1, dice_2]#, dice_3, dice_4]
    #model.compile(optimizer=Adam(lr=1e-4), loss=my_categorical_crossentropy, metrics=track_metrics)
    #model.compile(optimizer=Adam(lr=1e-6), loss=mc_generalized_dice_loss, metrics=track_metrics)
    model.compile(optimizer=Adam(lr=1e-4), loss=mc_tversky_loss, metrics=track_metrics)
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
