#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 12:29:33 2019

Use the Unet conv layer activations visualizer to run model on a particular
slice of a dataset and visualize activations of selected layers.
@author: ndahiya
"""

import SimpleITK as sitk
from unet_activations_visualizer import UNetActivationsVisualizer

dset_id = 'case_00004_dicom'
dset_name = '../results/unet_tversky_full_cropped/' + dset_id + '.nrrd'
dset_itk  = sitk.ReadImage(dset_name)
dset_arr  = sitk.GetArrayFromImage(dset_itk)
print(dset_arr.shape)
slice_num = 25
device = '/gpu:0'
model_name = 'unet_model_dilated_conv'
model_weights = '../results/models/unet_unet_tversky_full_cropped.hdf5'
num_classes = 3

visualizer = UNetActivationsVisualizer(slice_num=slice_num, dicom_arr=dset_arr,
                                       pretrained_weights_path=model_weights,
                                       unet_model_name=model_name,
                                       num_classes=num_classes,
                                       device=device)
visualizer.load_model()

#visualizer.plot_encoder_block(block_num=1, slice_num=20)
#visualizer.plot_encoder_block(block_num=2, slice_num=20)
#visualizer.plot_all_activations(slice_num=20)
#visualizer.plot_bottleneck_block(slice_num=20)

#visualizer.print_model_summary()
#visualizer.print_conv_layer_summary()
visualizer.create_activations(slice_num=20)
#visualizer.plot_all_activations()
visualizer.plot_decoder_block(block_num=4)