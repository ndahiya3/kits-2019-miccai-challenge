#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:43:53 2019

Running training with Unet model with given settings.
This file to be called via main.py which runs argument parsing and checking
before calling this train file.

@author: ndahiya
"""

import os
from models.unet_model_dilated_conv import unet_model_dilated_conv
from models.unet_model_deeper_dilated_conv import unet_model_deeper_dilated_conv
from models.unet_model_3d_tack_arch import unet_model_3d_tack_arch

from models.keras_generate_training_data import training_data_generator
from models.keras_generate_training_data import validation_data_generator
from models.keras_generate_training_data import data_generator_3d

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.callbacks import TensorBoard, ReduceLROnPlateau

import tensorflow as tf
import matplotlib.pyplot as plt

def train(train_dir, valid_dir, save_aug_dir, batch_size, model_save_path,
          pretrained_model_path, tb_log_dir, csv_log_path, num_classes,
          num_epochs, device, train_3d=False):
  # Model checkpoints/callbacks
  model_checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss',
                                     verbose=1, save_best_only=True)
  early_stopping   = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
  tensorboard      = TensorBoard(log_dir=tb_log_dir)
  csv_logger       = CSVLogger(csv_log_path, append=False, separator=';')
  lr_decay         = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                       patience=4, verbose=1)
  callback_list = [model_checkpoint, tensorboard,
                   csv_logger, early_stopping]#, lr_decay]
  # Figure out how many steps to run per epoch for train
  path, dirs, files = next(os.walk(os.path.join(train_dir, 'dicoms')))
  total_frames = len(files)

  if train_3d is False: # Augmentations for 2D network
    total_num_augmentations = 6
    train_steps_per_epoch = total_frames // batch_size
    train_steps_per_epoch = train_steps_per_epoch * total_num_augmentations
    train_steps_per_epoch = train_steps_per_epoch // 3
  else: # No augmentations for 3D network
    train_steps_per_epoch = total_frames // batch_size

  valid_steps_per_epoch = None
  if valid_dir is not None:
    path, dirs, files = next(os.walk(os.path.join(valid_dir, 'dicoms')))
    total_valid_frames = len(files)
    valid_steps_per_epoch = total_valid_frames // batch_size

  if train_3d is False:
    # List of image transformations to apply
    data_gen_args = dict(rotation_range=0.2,
                      width_shift_range=0.05,
                      height_shift_range=0.05,
                      shear_range=0.05,
                      zoom_range=0.05,
                      horizontal_flip=True,
                      fill_mode='nearest')
    # Training images generator
    img_generator = training_data_generator(train_dir, 'dicoms', 'masks',
                                            data_gen_args, num_classes,
                                            batch_size, save_aug_dir)

    # Validation images generator if valid_images dir path specified
    valid_generator = None
    if valid_dir is not None:
      valid_generator = validation_data_generator(valid_dir, 'dicoms', 'masks',
                                                  batch_size, num_classes)
    #model = unet_model_deeper_dilated_conv(pretrained_model_path, num_classes=num_classes)
    model = unet_model_dilated_conv(pretrained_model_path, num_classes=num_classes)
  else:
    img_generator = data_generator_3d(train_dir, 'dicoms', 'masks', batch_size)
    valid_generator = None
    if valid_dir is not None:
      valid_generator = data_generator_3d(valid_dir, 'dicoms', 'masks', batch_size)
    model = unet_model_3d_tack_arch(pretrained_model_path)

  # Run training on specified device
  with tf.device(device):
    history = model.fit_generator(img_generator,
                                  steps_per_epoch=train_steps_per_epoch,
                                  epochs=num_epochs,
                                  callbacks=callback_list,
                                  validation_data=valid_generator,
                                  validation_steps=valid_steps_per_epoch)
    # summarize history for accuracy
#    plt.plot(history.history['acc'])
#    plt.plot(history.history['val_acc'])
#    plt.title('model accuracy')
#    plt.ylabel('accuracy')
#    plt.xlabel('epoch')
#    plt.legend(['train', 'test'], loc='upper left')
#    plt.show()
#    # summarize history for loss
#    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
#    plt.title('model loss')
#    plt.ylabel('loss')
#    plt.xlabel('epoch')
#    plt.legend(['train', 'test'], loc='upper left')
#    plt.show()