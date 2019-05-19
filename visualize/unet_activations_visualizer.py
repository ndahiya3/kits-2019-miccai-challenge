#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 11:26:42 2019

Class to visualize the activations of the convolutional layer of a trained UNet
model.
@author: ndahiya
"""

import sys, os
sys.path.insert(0, os.path.abspath('..'))

from models.unet_model_dilated_conv import unet_model_dilated_conv
from keras.models import Model
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

class UNetActivationsVisualizer:
  """
  Class for visualizing activations of a trained 2D UNet for a particular
  slice.
  """
  def __init__(self, slice_num=0, dicom_arr=None,
               pretrained_weights_path=None,
               unet_model_name=None,
               num_classes=1,
               device='/gpu:0'):

    self.slice_num = slice_num
    self.img_arr = dicom_arr
    self.model_weights = pretrained_weights_path
    self.model_name = unet_model_name
    self.device = device
    self.num_classes = num_classes
    self.model_loaded = False
    self.all_layers_extracted = False
    self.conv_layers_extracted = False
    self.activation_model_created = False
    self.activations_created = False
    self.num_encoders = 0
    self.num_decoders = 0
    self.num_bottleneck_layers = 0

    self.is_model_ready = self.model_ready()

  def set_data_array(self, dicom_arr):
    """
    Set the input dicom array
    """
    self.img_arr = dicom_arr
    self.is_model_ready = self.model_ready()

  def set_model_weights_path(self, pretrained_weights_path):
    """
    Set the name/path of the pretrained model weights. These weights will be
    loaded and used to run inference needed to visualize activations.
    """
    self.model_weights = pretrained_weights_path
    self.is_model_ready = self.model_ready()

  def set_model_name(self, unet_model_name):
    """
    Set the name of the model being used. There are a few different varieties
    of Unet models.
    """
    self.model_name = unet_model_name
    self.is_model_ready = self.model_ready()

  def set_slice_num(self, slice_num):
    """
    Set the slice num within the dicom array to use for inference. Different
    slice locations will give different visual activations based on location
    of the slice within anatomy.
    """
    self.slice_num = slice_num
    self.is_model_ready = self.model_ready()

  def set_device(self, device):
    """
    Set which device to use for inference, e.g. /cpu:0, /gpu:0, /gpu:1 etc.
    """
    self.device = device

  def set_num_classes(self, num_classes=1):
    """
    Set number of output classes in the model.
    """
    self.num_classes = num_classes
    self.is_model_ready = self.model_ready()

  def model_ready(self):
    """
    Check if all parameters are ready to load/run model.
    """
    if self.img_arr is None or self.model_name is None or self.model_weights is None:
      return False
    else:
      return True

  def load_model(self):
    """ Load the model with pretrained weights.
    """
    if self.is_model_ready is False:
      print("=================== Error ====================")
      print("Cannot load model. All parameters needed to load model not available.")
      print("=================== Error ====================")

      self.model_loaded = False
    else:
      with tf.device(self.device):
        self.model = unet_model_dilated_conv(pretrained_weights=self.model_weights,
                                           num_classes=self.num_classes, input_size=(1,256,256))
      self.model_loaded = True

  def print_model_summary(self):
    """
    Print the summary of the loaded model.
    """
    if self.model_loaded is False:
      print("=================== Error ====================")
      print("Model not yet loaded. Please load model first.")
      print("=================== Error ====================")
    else:
      self.model.summary()

  def extract_layers(self):
    """
    Internal function to extract the conv2D layers out of the model. Used to
    initialize new model which will output the desired activations after
    running inference.
    """
    # Extract layer outputs and names, ignore input layer
    if self.model_loaded is False:
      print("=================== Error ====================")
      print("Model not yet loaded. Please load model first.")
      print("=================== Error ====================")
      return False
    else:
      self.layer_outputs = [layer.output for layer in self.model.layers[1:]]
      self.layer_names   = [layer.name for layer in self.model.layers[1:]]
      self.all_layers_extracted = True

      return True

  def identify_conv_layers(self):
    """
    Identify the convolutional layers out of all layers of the model.
    Ignore the last 1x1 output convolution layer.
    """
    if self.all_layers_extracted is False:
      if not self.extract_layers():
        return False

    self.conv_layers = []
    self.conv_layer_names = []

    for layer, layer_name in zip(self.layer_outputs, self.layer_names):
      if 'conv2d' in layer_name:
        self.conv_layers.append(layer)
        self.conv_layer_names.append(layer_name)

    # Ignore last 1x1 convolution
    self.conv_layer_names = self.conv_layer_names[:-1]
    self.conv_layers = self.conv_layers[:-1]

    self.conv_layers_extracted = True
    self.print_conv_layer_summary()

    return True

  def print_conv_layer_summary(self):
    """
    Print layer names and shapes of only convolutional layers.
    """
    if self.conv_layers_extracted is False:
      if not self.identify_conv_layers():
        return

    print("=====================================")
    print("Convolutional Layers Summary in Model")
    print("=====================================")
    for layer, layer_name in zip(self.conv_layers, self.conv_layer_names):
      print(layer_name, layer.shape)
    print("=====================================")

    num_conv_layers = len(self.conv_layers) // 2 # Each layer has 2 convs
    num_encoder_blocks = num_conv_layers // 2
    num_decoder_blocks = num_encoder_blocks
    num_bottleneck_layers = 1

    self.num_encoders = num_encoder_blocks
    self.num_decoders = num_decoder_blocks
    self.num_bottleneck_layers = num_bottleneck_layers

    print("Model has:")
    print("{} encoder blocks:".format(num_encoder_blocks))
    print("{} decoder blocks:".format(num_decoder_blocks))
    print("with {} bottleneck layers".format(num_bottleneck_layers))

  def create_activation_model(self):
    """
    Create activation model used to run inference such that activations of
    selected layers are output, which can then be visualized.
    """
    if self.conv_layers_extracted is False:
      if not self.identify_conv_layers():
        return False

    self.activation_model = Model(inputs=self.model.input, outputs=self.conv_layers)
    self.activation_model_created = True
    return True

  def create_activations(self, slice_num=None):
    """
    Run the model on a given slice and extract the activations of convolutional
    layers.
    """
    if self.activation_model_created is False:
      if not self.create_activation_model():
        return False

    if slice_num is not None:
      self.slice_num = slice_num

    input_img = self.img_arr[self.slice_num]
    input_img = np.expand_dims(input_img, axis=0)
    input_img = np.expand_dims(input_img, axis=0).astype(np.float32)

    self.conv_layer_activations = self.activation_model.predict(input_img)
    self.activations_created = True

    return True

  def plot_encoder_block(self, block_num=1, slice_num=None):
    """
    Plot activations of a particular encoder block.
    Arguments:
      block_num: integer encoder block number from [1 num_encoder_blocks]
    """

    if not self.__check_before_plotting_activations(slice_num):
      return

    block_num -= 1 # Make 0 indexed
    if block_num >= self.num_encoders or block_num < 0:
      print("Encoder block number out of range. Plotting encoder block num 1...")
      block_num = 0
    else:
      print("Plotting Encoder block number: {}".format(block_num+1))

    # Ready to plot
    # =======================
    # Convert block number to convolutional layer numbers
    # Each encoder block has 2 conv layers
    conv_layer_idx = block_num*2

    # Extract only layer activations/names for required encoder block
    layer_names = []
    layer_names.append(self.conv_layer_names[conv_layer_idx])
    layer_names.append(self.conv_layer_names[conv_layer_idx + 1])

    layer_activations = []
    layer_activations.append(self.conv_layer_activations[conv_layer_idx])
    layer_activations.append(self.conv_layer_activations[conv_layer_idx + 1])

    title = "Encoder Block # {}: ".format(block_num)
    self.__plot_layers(layer_names, layer_activations, title)

  def plot_decoder_block(self, block_num=1, slice_num=None):
    """
    Plot activations of a particular decoder block.
    Arguments:
      block_num: integer decoder block number from [1 num_decoder_blocks]
    """

    if not self.__check_before_plotting_activations(slice_num):
      return

    block_num -= 1 # Make 0 indexed
    if block_num >= self.num_decoders or block_num < 0:
      print("Decoder block number out of range. Plotting decoder block num 1...")
      block_num = 0
    else:
      print("Plotting Decoder block number: {}".format(block_num+1))

    # Ready to plot
    # =======================
    # Convert block number to convolutional layer numbers
    # Each decoder block has 2 conv layers
    conv_layer_idx = len(self.conv_layer_activations) - block_num*2 - 2

    # Extract only layer activations/names for required encoder block
    layer_names = []
    layer_names.append(self.conv_layer_names[conv_layer_idx])
    layer_names.append(self.conv_layer_names[conv_layer_idx + 1])

    layer_activations = []
    layer_activations.append(self.conv_layer_activations[conv_layer_idx])
    layer_activations.append(self.conv_layer_activations[conv_layer_idx + 1])

    title = "Decoder Block # {}: ".format(block_num)
    self.__plot_layers(layer_names, layer_activations, title)

  def plot_bottleneck_block(self, slice_num=None):
    """
    Plot activations of the Bottleneck layer. Bottleneck layer is last
    encode block with the input image converted to least resolution
    before starting to decode/up sample. It detects the coarest features at
    lowest resolution.
    """
    if not self.__check_before_plotting_activations(slice_num):
      return

    conv_layer_idx = self.num_encoders*2
    # Extract only layer activations/names for required encoder block
    layer_names = []
    layer_names.append(self.conv_layer_names[conv_layer_idx])
    layer_names.append(self.conv_layer_names[conv_layer_idx + 1])

    layer_activations = []
    layer_activations.append(self.conv_layer_activations[conv_layer_idx])
    layer_activations.append(self.conv_layer_activations[conv_layer_idx + 1])

    title = "Bottleneck Layer: "
    self.__plot_layers(layer_names, layer_activations, title)

  def plot_all_activations(self, slice_num=None):
    """
    Plot the activations of all convolution layers
    """
    if not self.__check_before_plotting_activations(slice_num):
      return

    self.__plot_layers(self.conv_layer_names, self.conv_layer_activations)

  def __plot_layers(self, layer_names, layer_activations, title=None):
    """
    Plot layer provided layer names and layer activations in two lists
    """
    images_per_row = 16

    for layer_name, layer_activation in zip(layer_names, layer_activations): # Displays the feature maps
      n_features = layer_activation.shape[1] # Number of features in the feature map
      size = layer_activation.shape[2] #The feature map has shape (1, n_features, size, size).
      n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
      display_grid = np.zeros((size * n_cols, images_per_row * size))
      for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
          channel_image = layer_activation[0,col * images_per_row + row,:,:]
          channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
          channel_image /= channel_image.std() + 1e-6
          channel_image *= 64
          channel_image += 128
          channel_image = np.clip(channel_image, 0, 255).astype('uint8')
          display_grid[col * size : (col + 1) * size, # Displays the grid
                       row * size : (row + 1) * size] = channel_image
      scale = 1. / size
      plt.figure(figsize=(scale * display_grid.shape[1],
                          scale * display_grid.shape[0]))
      if title is not None:
        plt_title = title + layer_name
      else:
        plt_title = layer_name
      plt.title(plt_title)
      plt.grid(False)
      plt.imshow(display_grid, aspect='auto', cmap='viridis')

    plt.show()

  def __check_before_plotting_activations(self, slice_num=None):
    """
    Private function to check if model is loaded, layers extracted,
    activation model created and activations created for a given slice
    number before trying to plot layer activations.
    """
    if slice_num is not None:
      if not self.create_activations(slice_num):
        return False # Means that model not loaded yet
      else:
        pass
    else:
      if not self.activations_created:
        if not self.create_activations():
          return False # Means that model not loaded yet
    return True






































