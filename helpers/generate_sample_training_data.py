#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:32:53 2019

Split MRI datasets and corresponding masks into individual frames
for training with Keras. It is possible to write custom training data
generators but as a first experiment its easier to just split the 3D
datasets into individual frames and directly use Keras' inbuilt data
augumentation/generator functionality.

For CNN-3D we need to sample sub-volumes along the boundary of true/predicted
volume.

@author: ndahiya
"""

import SimpleITK as sitk
import numpy as np
import cv2
import os
from scipy import ndimage as ndi
from .utilities import get_dataset_ids

def extract_cropped_kidney_data(dset_ids_file, in_dir_path,
                 out_dicom_dir, out_masks_dir, num_dsets_to_extract,
                 convert_masks_to_binary=True,classes_to_keep=[],
                 extract_only_with_foreground=False):
  """
  Function that actually extracts data.
  Parameters
  ----------
  oai_ids_file : string
                 Filename containing dataset ids to extract
  in_dir_path : string
                Directory containing *.nrrd format dicoms with corresp. masks
  out_dicom_dir : string
                  Directory to output dicom frames
  out_masks_dir : string
                  Directory to output masks frames
  num_dsets_to_extract : integer
                          Number of datasets to extract
  convert_masks_to_binary : boolean
                            Convert multi class label masks to 0/1
  class_to_keep : integer
                  convert multi class to binary keeping class_to_keep as
                  foreground and rest as background. default = 0 is to keep
                  whole knee as foreground
  extract_only_with_foreground : boolean
                                 Extract only frames which have some foreground

  """

  id_list = get_dataset_ids(dset_ids_file, num_dsets_to_extract)
  if num_dsets_to_extract > len(id_list):
    print("Number of datasets to extract more than available. Extracting all "
          "available")
    num_dsets_to_extract = len(id_list)

  dset_stats_file = open('cropped_dsets_stats_file.txt','a+')
  for i in range(num_dsets_to_extract):
    curr_id = id_list[i]
    print("Extracting {}: {} of {}".format(curr_id, i+1, num_dsets_to_extract))
    dicom_file = os.path.join(in_dir_path, curr_id + '_dicom.nrrd')
    mask_file  = os.path.join(in_dir_path, curr_id + '_seg_mask.nrrd')

    dicom_image_3d = sitk.ReadImage(dicom_file)
    mask_image_3d  = sitk.ReadImage(mask_file)

    #print('DICOM size:\t', dicom_image_3d.GetSize())
    np_arr_dicom = sitk.GetArrayFromImage(dicom_image_3d)
    np_arr_mask_orig  = sitk.GetArrayFromImage(mask_image_3d)
    np_arr_mask = np.zeros_like(np_arr_mask_orig)

    if convert_masks_to_binary is True:
      for class_to_keep in classes_to_keep:
        np_arr_mask[np.where(np_arr_mask_orig == np.uint(class_to_keep))] = 1
    else: # Classes to keep are saved with labels 1, 2, 3 ...
      for idx, class_to_keep in enumerate(classes_to_keep):
        np_arr_mask[np.where(np_arr_mask_orig == np.uint(class_to_keep))] = idx + 1

    min_val = np_arr_dicom.min()
    print('\tDataset Size:', np_arr_dicom.shape)
    print('\tMin image value: ', min_val)
    print('\tMax image value: ', np_arr_dicom.max())

    # Extra prep for SN Data
    if min_val < 0:
      np_arr_dicom = np_arr_dicom - min_val # Shift data to make min == 0
    np_arr_dicom = np_arr_dicom.astype(np.uint16) # Signed int16 to uint

    # Save individual dicom and mask frames
    extracted_frames = 0
    nnz_frames = 0 # Number of frames with non-blank mask
    foreground_only_idx = 0
    nnz_mismatch = 0
    for frame in range(np_arr_dicom.shape[2]):
        curr_dicom_frame_path = os.path.join(out_dicom_dir, curr_id + '.{}'.format(frame) + '.png')
        curr_mask_frame_path  = os.path.join(out_masks_dir, curr_id + '.{}'.format(frame) + '.png')

        #print(curr_dicom_frame_path, curr_mask_frame_path)
        nnz = np.count_nonzero(np_arr_mask[:,:,frame])
        if nnz > 0:
          nnz_frames += 1
        if extract_only_with_foreground == True:
          if nnz != 0:
            curr_dicom_frame_path = os.path.join(out_dicom_dir, curr_id + '.{}'.format(foreground_only_idx) + '.png')
            curr_mask_frame_path  = os.path.join(out_masks_dir, curr_id + '.{}'.format(foreground_only_idx) + '.png')
            dicom_frame = crop_frame(np_arr_dicom[:,:,frame].T)
            mask_frame  = crop_frame(np_arr_mask[:,:,frame].T)
            cv2.imwrite(curr_dicom_frame_path, dicom_frame)
            cv2.imwrite(curr_mask_frame_path, mask_frame)
            extracted_frames += 1
            foreground_only_idx += 1
        else:
          dicom_frame = crop_frame(np_arr_dicom[:,:,frame].T)
          mask_frame  = crop_frame(np_arr_mask[:,:,frame].T)
          cv2.imwrite(curr_dicom_frame_path, dicom_frame)
          cv2.imwrite(curr_mask_frame_path, mask_frame)
          nnz_cropped = np.count_nonzero(mask_frame)
          if (nnz != nnz_cropped):
            nnz_mismatch += 1
          extracted_frames += 1
    print("\tExtracted {} frames".format(extracted_frames))
    print("{}: ({},{},{})\t {}/{}".format(curr_id, np_arr_dicom.shape[0],
          np_arr_dicom.shape[1], np_arr_dicom.shape[2], nnz_mismatch, nnz_frames), file=dset_stats_file)
  dset_stats_file.close()

def crop_frame(frame):
  """
  Crop a kidney frame. Crop region is 256x256 around image center.
  """
  #cropped = frame[256-192:256+192,256-192:256+192]
  cropped = frame[256-128+25:256+128+25,256-128:256+128]
  #print(cropped.shape)
  return cropped

def get_boundary(in_mask, num_volumes):
  """
  Select desired number of locations on input mask boundary.
  # Arguments:
      in_mask: numpy array of input mask, in_mask.shape = [slices, H, W]
      num_volumes: number of sub-volumes to sample
  # Returns:
      2D array of indexes [one per row] of selected sub-volume centers
      on mask boundary.
  """

  # Extract boundary of the input mask [binary XOR]
  connectivity_struc_elem = ndi.generate_binary_structure(3, 3)
  border = in_mask ^ ndi.binary_erosion(in_mask,
                                        structure=connectivity_struc_elem,
                                        iterations=1)
  return border

def get_subvolumes_centers(in_mask, num_volumes):
  """
  Select desired number of locations on input mask boundary.
  # Arguments:
      in_mask: numpy array of input mask, in_mask.shape = [slices, H, W]
      num_volumes: number of sub-volumes to sample
  # Returns:
      2D array of indexes [one per row] of selected sub-volume centers
      on mask boundary.
  """

  # Extract boundary of the input mask [binary XOR]
  connectivity_struc_elem = ndi.generate_binary_structure(3, 3)
  border = in_mask ^ ndi.binary_erosion(in_mask,
                                        structure=connectivity_struc_elem,
                                        iterations=1)
  z,y,x = np.nonzero(border)
  boundary_indexes = np.vstack((z,y,x)).T # Indexes of boundary points, one per row
  # Randomly select num_volumes points on boundary
  selected_boundary_points = np.random.choice(boundary_indexes.shape[0], # return random row numbers
                                              num_volumes, replace=False)
  selected_boundary_indexes = boundary_indexes[selected_boundary_points]

  return selected_boundary_indexes

def get_subvolume_idx_ranges(curr_point, max_dim, desired_dims):
  """
  Get the index range [min max] around the sub-volume center. Center is shifted
  to keep the sampled sub-volume within dataset dimensions.
  # Arguments:
      curr_point: Coordinate, scalar, one of x, y, or z, of sub-volume center
      max_dim: Max dimension of the underlying dataset, cannot exceed this
      desired_dims: Desired sub-volume dimension around the center point
  # Returns:
      Tuple of minimum and maximum index
  """
  min_idx = curr_point - (desired_dims//2 - 1)
  max_idx = curr_point + (desired_dims//2 + 1)

  if min_idx < 0: # Shift right
    max_idx += np.abs(min_idx)
    min_idx = 0

  if max_idx > max_dim: # Shift left
    min_idx -= max_idx - max_dim
    max_idx = max_dim

  return (min_idx, max_idx)

def extract_3d_subvolumes(dset_ids_file, in_dir_path, out_dicom_dir,
                          out_masks_dir, num_dsets_to_extract,
                          num_subvolumes_to_extract, desired_dims,
                          class_to_keep=1):
  """
  Function that actually extracts data.
  Parameters
  ----------
  oai_ids_file : string
                 Filename containing dataset ids to extract
  in_dir_path : string
                Directory containing *.nrrd format dicoms with corresp. masks
  out_dicom_dir : string
                  Directory to output dicom frames
  out_masks_dir : string
                  Directory to output masks frames
  num_dsets_to_extract : integer
                          Number of datasets to extract
  num_subvolumes_to_extract : integer
                              Number of subvolumes to extract from each dataset
  desired_dims : 3-tuple of integers
                 Desired subvolume dimensions
  class_to_keep : integer
                  If multiple classes present in masks, only keep this class label
  """
  id_list = get_dataset_ids(dset_ids_file, num_dsets_to_extract)
  if num_dsets_to_extract > len(id_list):
    print("Number of datasets to extract more than available. Extracting all."
          "available")
    num_dsets_to_extract = len(id_list)

  for i in range(num_dsets_to_extract):
    curr_id = id_list[i]
    print("Sampling subvolumes {}: {} of {}".format(curr_id, i+1, num_dsets_to_extract))
    dicom_file = os.path.join(in_dir_path, curr_id + '.nrrd')
    mask_file  = os.path.join(in_dir_path, curr_id + '_seg_mask.nrrd')

    dicom_image_3d = sitk.ReadImage(dicom_file)
    mask_image_3d  = sitk.ReadImage(mask_file)

    #print('DICOM size:\t', dicom_image_3d.GetSize())
    np_arr_dicom = sitk.GetArrayFromImage(dicom_image_3d)
    np_arr_mask_orig  = sitk.GetArrayFromImage(mask_image_3d)
    np_arr_mask = np.zeros_like(np_arr_mask_orig)
    # Keep desired class
    np_arr_mask[np.where(np_arr_mask_orig == int(class_to_keep))] = 1

    # Extract and save sub-volumes
    subvolume_centers = get_subvolumes_centers(np_arr_mask, num_subvolumes_to_extract)
    dims = np_arr_dicom.shape
    idx = 0
    for center in subvolume_centers:
      z_min, z_max = get_subvolume_idx_ranges(center[0], dims[0], desired_dims[0])
      y_min, y_max = get_subvolume_idx_ranges(center[1], dims[1], desired_dims[1])
      x_min, x_max = get_subvolume_idx_ranges(center[2], dims[2], desired_dims[2])

      subvolume_dims = (z_max-z_min, y_max-y_min, x_max-x_min)
      if (subvolume_dims != desired_dims):
        print("skipping")
        continue
      # Get dicom and mask subvolumes
      dicom_subvolume = np_arr_dicom[z_min:z_max, y_min:y_max, x_min:x_max]
      mask_subvolume  = np_arr_mask[z_min:z_max, y_min:y_max, x_min:x_max]

      # Save dicom and mask subvolumes
      dicom_subvolume_itk = sitk.GetImageFromArray(dicom_subvolume)
      mask_subvolume_itk  = sitk.GetImageFromArray(mask_subvolume)

      curr_dicom_path = os.path.join(out_dicom_dir, curr_id + '.{}'.format(idx) + '.nrrd')
      curr_mask_path  = os.path.join(out_masks_dir, curr_id + '.{}'.format(idx) + '.nrrd')

      sitk.WriteImage(dicom_subvolume_itk, curr_dicom_path)
      sitk.WriteImage(mask_subvolume_itk, curr_mask_path)

      idx += 1

def point2str(point, precision=2):
  """
  Format a point for printing
  """
  return ' '.join(format(c, '.{0}f'.format(precision)) for c in point)

def extract_data(dset_ids_file, in_dir_path,
                 out_dicom_dir, out_masks_dir, num_dsets_to_extract,
                 convert_masks_to_binary=True,classes_to_keep=[],
                 extract_only_with_foreground=False):
  """
  Function that actually extracts data.
  Parameters
  ----------
  oai_ids_file : string
                 Filename containing dataset ids to extract
  in_dir_path : string
                Directory containing *.nrrd format dicoms with corresp. masks
  out_dicom_dir : string
                  Directory to output dicom frames
  out_masks_dir : string
                  Directory to output masks frames
  num_dsets_to_extract : integer
                          Number of datasets to extract
  convert_masks_to_binary : boolean
                            Convert multi class label masks to 0/1
  class_to_keep : integer
                  convert multi class to binary keeping class_to_keep as
                  foreground and rest as background. default = 0 is to keep
                  whole knee as foreground
  extract_only_with_foreground : boolean
                                 Extract only frames which have some foreground

  """

  id_list = get_dataset_ids(dset_ids_file, num_dsets_to_extract)
  if num_dsets_to_extract > len(id_list):
    print("Number of datasets to extract more than available. Extracting all "
          "available")
    num_dsets_to_extract = len(id_list)

  dset_stats_file = open('dsets_stats_file.txt','a+')
  for i in range(num_dsets_to_extract):
    curr_id = id_list[i]
    print("Extracting {}: {} of {}".format(curr_id, i+1, num_dsets_to_extract))
    dicom_file = os.path.join(in_dir_path, curr_id + '_dicom.nii.gz')
    mask_file  = os.path.join(in_dir_path, curr_id + '_seg_mask.nii.gz')

    dicom_image_3d = sitk.ReadImage(dicom_file)
    mask_image_3d  = sitk.ReadImage(mask_file)

    #print('DICOM size:\t', dicom_image_3d.GetSize())
    np_arr_dicom = sitk.GetArrayFromImage(dicom_image_3d)
    np_arr_mask_orig  = sitk.GetArrayFromImage(mask_image_3d)
    np_arr_mask = np.zeros_like(np_arr_mask_orig)

    if convert_masks_to_binary is True:
      for class_to_keep in classes_to_keep:
        np_arr_mask[np.where(np_arr_mask_orig == np.uint(class_to_keep))] = 1
    else: # Classes to keep are saved with labels 1, 2, 3 ...
      for idx, class_to_keep in enumerate(classes_to_keep):
        np_arr_mask[np.where(np_arr_mask_orig == np.uint(class_to_keep))] = idx + 1

    # min_val = np_arr_dicom.min()
    # print('\tDataset Size:', np_arr_dicom.shape)
    # print('\tMin image value: ', min_val)
    # print('\tMax image value: ', np_arr_dicom.max())
    #
    # # Extra prep for SN Data
    # if min_val < 0:
    #   np_arr_dicom = np_arr_dicom - min_val # Shift data to make min == 0
    # np_arr_dicom = np_arr_dicom.astype(np.uint16) # Signed int16 to uint

    # Save individual dicom and mask frames
    extracted_frames = 0
    nnz_frames = 0 # Number of frames with non-blank mask
    for frame in range(np_arr_dicom.shape[2]):
        curr_dicom_frame_path = os.path.join(out_dicom_dir, curr_id + '.{}'.format(frame) + '.npz')
        curr_mask_frame_path  = os.path.join(out_masks_dir, curr_id + '.{}'.format(frame) + '.npz')

        #print(curr_dicom_frame_path, curr_mask_frame_path)
        nnz = np.count_nonzero(np_arr_mask[:,:,frame])
        if nnz > 0:
          nnz_frames += 1
        if extract_only_with_foreground == True and nnz == 0:
          continue

        np.savez(curr_dicom_frame_path, A=np_arr_dicom[:, :, frame].T)
        np.savez(curr_mask_frame_path, A=np_arr_mask[:, :, frame].T)
        extracted_frames += 1

    print("\tExtracted {} frames".format(extracted_frames))
    print("{}: ({},{},{})\t {}/{}".format(curr_id, np_arr_dicom.shape[0],
          np_arr_dicom.shape[1], np_arr_dicom.shape[2], nnz_frames, np_arr_dicom.shape[2]), file=dset_stats_file)
  dset_stats_file.close()