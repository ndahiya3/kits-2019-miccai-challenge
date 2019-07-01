#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:12:41 2019
Enet model. A small model for segmentation.
Ref: https://arxiv.org/pdf/1606.02147.pdf
@author: ndahiya
"""
import sys,os
sys.path.insert(0, os.path.abspath('.'))

import keras.backend as K
K.set_image_data_format('channels_last')

from models.enet_encoder import build_encoder
from models.enet_decoder import build_decoder
from keras.layers import Input, Activation
from keras.optimizers import Adam
from keras.models import Model
from models.loss_metrics import dice_1, dice_2
from models.loss_functions import mc_tversky_loss, dice_coef_loss, dice_coef
from models.loss_functions import my_categorical_crossentropy

def enet_model(pretrained_weights=None,input_shape=(512,512,1), num_classes=2):

  input_img = Input(shape=input_shape)
  encoder   = build_encoder(input_img)
  decoder   = build_decoder(encoder, num_classes=num_classes)

  if num_classes > 1:
    output = Activation('softmax')(decoder)
    model = Model(inputs=input_img, outputs=output)
    track_metrics = [dice_1]
    model.compile(optimizer=Adam(lr=1e-5), loss=mc_tversky_loss, metrics=track_metrics)
    #model.compile(optimizer=Adam(lr=1e-4), loss=my_categorical_crossentropy, metrics=track_metrics)
  else:
    output = Activation('sigmoid')(decoder)
    model = Model(inputs=input_img, outputs=output)
    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=['accuracy', dice_coef])

  if (pretrained_weights is not None):
    model.load_weights(pretrained_weights)
  return model