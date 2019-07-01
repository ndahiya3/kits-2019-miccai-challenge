#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 00:09:18 2019

@author: ndahiya
"""

import numpy as np
import keras
import keras.backend as K
from os import path
import sys
sys.path.insert(0, path.abspath('..'))
from helpers.utilities import get_files_list_dir

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, base_dir, dicom_folder, masks_folder, batch_size=32,
                 n_classes=10, shuffle=True, mean_normalize=True, n_channels=1, dim = (512,512)):
        'Initialization'
        self.masks_dir = path.abspath(path.join(base_dir, masks_folder))
        self.dicom_dir   = path.abspath(path.join(base_dir, dicom_folder))
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.n_channels = n_channels
        self.dim = (512,512)

        self.mean_normalize = mean_normalize
        files_list  = get_files_list_dir(self.dicom_dir)
        datasets_names = []
        for file in files_list:
          dset_name = file.split('/')[-1]
          datasets_names.append(dset_name)
        print("Found {} files.".format(len(datasets_names)))
        self.list_IDs = datasets_names
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X = np.empty((self.batch_size, self.n_channels, *self.dim))
        y = np.empty((self.batch_size, self.n_channels, *self.dim), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            dicom_path = path.join(self.dicom_dir, ID)
            mask_path  = path.join(self.masks_dir, ID)
            img = np.load(dicom_path)['A']
            if img.shape[0] != 512 or img.shape[1] != 512:
              print(dicom_path)
            X[i,] = np.expand_dims(np.load(dicom_path)['A'], axis=0)

            # Store class
            y[i] = np.expand_dims(np.load(mask_path)['A'], axis=0)
        if (self.n_classes > 1):
          y  = self.__convert_masks_one_hot(y, self.n_classes)
        if self.mean_normalize:
          X = self.__batch_mean_normalize(X)
        #print("Data shape: ", X.shape)
        #print("Masks shape: " , y.shape)

        if K.image_data_format() == 'channels_last':
          X = np.moveaxis(X, 1, -1)
          y = np.squeeze(y, axis=1)
        #print("Data shape: ", X.shape)
        return X, y

    def __convert_masks_one_hot(self, masks, num_classes):
        # Convert multi label masks to one hot format
        #print('before: ', masks.shape)
        masks =  keras.utils.to_categorical(masks, num_classes=num_classes)#, dtype='uint8')
        #masks = to_categorical(masks, num_classes=None)
        if K.image_data_format() == 'channels_first':
          masks = np.moveaxis(masks, -1, 1)
          masks = np.squeeze(masks, axis=2)
        else:
          pass
        #print('after: ', masks.shape)
        return masks

    def __batch_mean_normalize(self, batch_data):
        # Batch normalize a batch of 2D slices
        batch_mean = batch_data.mean()
        batch_data = batch_data - batch_mean
        batch_data = batch_data/(batch_data.var()**0.5)

        return batch_data