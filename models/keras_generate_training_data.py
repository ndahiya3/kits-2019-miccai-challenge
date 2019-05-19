#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:48:13 2019
Generate augmented training data using Keras flow from directory where
training DICOM and mask frames are saved.
@author: ndahiya
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from os import path
import sys
sys.path.insert(0, path.abspath('..'))
from helpers.utilities import get_files_list_dir
from helpers.utilities import read_arr_from_itk

def data_generator_3d(base_dir, dicom_folder, masks_folder, batch_size=1):
  # Training/validation data generator for running 3D Unet using small
  # subsampled volumes
  masks_dir = path.abspath(path.join(base_dir, masks_folder))
  dicom_dir   = path.abspath(path.join(base_dir, dicom_folder))

  files_list  = get_files_list_dir(dicom_dir)

  datasets_names = []
  for file in files_list:
    dset_name = file.split('/')[-1]
    datasets_names.append(dset_name)

  print("Found {} 3D volumes.".format(len(datasets_names)))
  while True:
    batch_dsets = np.random.choice(datasets_names, batch_size, replace=False)
    # Read in each dataset in current batch
    batch_dicoms = []
    batch_masks  = []
    for dset in batch_dsets:
      dicom_path = path.join(dicom_dir, dset)
      mask_path  = path.join(masks_dir, dset)
      dicom_arr  = read_arr_from_itk(dicom_path)
      mask_arr   = read_arr_from_itk(mask_path)

      dicom_arr = np.expand_dims(dicom_arr, axis=0) # Model needs extra dim for classes
      mask_arr  = np.expand_dims(mask_arr, axis=0)

      batch_dicoms += [dicom_arr]
      batch_masks  += [mask_arr]

    # Yield a batch of dicom/mask data
    #yield(batch_dsets)
    yield(np.array(batch_dicoms), np.array(batch_masks))

def convert_masks_one_hot(masks, num_classes):
  # Convert multi label masks to one hot format
  #print('before: ', masks.shape)
  masks = to_categorical(masks, num_classes=num_classes)#, dtype='uint8')
  #masks = to_categorical(masks, num_classes=None)
  masks = np.moveaxis(masks, -1, 1)
  masks = np.squeeze(masks, axis=2)
  #print('after: ', masks.shape)
  return masks

def scale_data(dicom):
  max_val = dicom.max()
  if max_val:
    dicom = dicom / max_val
  return dicom

def training_data_generator(train_path, dicom_folder, masks_folder,\
                            augment_dict, num_classes, batch_size=1,
                             save_aug_dir=None):
  dicom_datagen = ImageDataGenerator(**augment_dict)
  masks_datagen = ImageDataGenerator(**augment_dict)

  # Same seed for dicom and mask generator to ensure same transformations
  # are applied to both
  seed = 1

  dicom_generator = dicom_datagen.flow_from_directory(train_path,
                                    classes = [dicom_folder],
                                    target_size=(256,256),
                                    color_mode='grayscale',
                                    class_mode=None,
                                    batch_size=batch_size,
                                    save_to_dir=save_aug_dir,
                                    seed=seed)
  masks_generator = masks_datagen.flow_from_directory(train_path,
                                    classes = [masks_folder],
                                    target_size=(256,256),
                                    color_mode='grayscale',
                                    class_mode=None,
                                    batch_size=batch_size,
                                    seed=seed)
  train_generator = zip(dicom_generator, masks_generator)
  for (dicom, mask) in train_generator:
    dicom = scale_data(dicom)
    #print(dicom.shape, mask.shape)
    if (num_classes > 1):
      mask  = convert_masks_one_hot(mask, num_classes)
    yield (dicom, mask)

def validation_data_generator(valid_path, dicom_folder, masks_folder,
                              batch_size=1, num_classes=1):
  dicom_datagen = ImageDataGenerator()
  masks_datagen = ImageDataGenerator()

  seed = 1
  dicom_generator = dicom_datagen.flow_from_directory(valid_path,
                                    classes=[dicom_folder],
                                    target_size=(256,256),
                                    color_mode='grayscale',
                                    class_mode=None,
                                    batch_size=batch_size,
                                    seed=seed)
  masks_generator = masks_datagen.flow_from_directory(valid_path,
                                    classes=[masks_folder],
                                    target_size=(256,256),
                                    color_mode='grayscale',
                                    class_mode=None,
                                    batch_size=batch_size,
                                    seed=seed)
  valid_generator = zip(dicom_generator, masks_generator)
  for (dicom, mask) in valid_generator:
    dicom = scale_data(dicom)
    if (num_classes > 1):
      mask  = convert_masks_one_hot(mask, num_classes)
    yield (dicom, mask)
