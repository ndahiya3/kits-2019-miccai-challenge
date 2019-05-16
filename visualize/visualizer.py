#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:17:22 2019
Browse slices of MRI 3D volume using matplotlib event handler api. Overlay
true/predicted masks over MRI and predicted mask as contour over true mask.

@author: ndahiya
"""
import matplotlib.pyplot as plt
import numpy as np
# Global variable
txt = None

def multi_slice_viewer(image, title=None):
  # Simply display 3D image slice by slice
  print("Please use j/k keys to browse forward/backward through volume")
  remove_keymap_conflicts({'j', 'k'})
  fig, ax1 = plt.subplots(1, 1, sharex=True, sharey=True)
  # Store the Input volume
  ax1.volume = image
  # Start showing mri with middle slice
  ax1.index = image.shape[0] // 2
  ax1.imshow(image[ax1.index], cmap=plt.cm.gray)
  ax1.axis('off')
  ax1.set_title('Input Image')
  fig.canvas.mpl_connect('key_press_event', process_key_simple_viewer)
  fig.suptitle(title)
  plt.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.3)
  global txt
  txt = plt.figtext(.6, .9, "Showing Slice # {:<0d}".format(ax1.index))
  #plt.tight_layout()
  plt.show()
  
def multi_slice_overlay_viewer(mri, true_mask, pred_mask, title=None):
  # Overlay predicted mask as contour over MRI and True mask in separate
  # sub-plots
  print("Please use j/k keys to browse forward/backward through volume")
  remove_keymap_conflicts({'j', 'k'})
  fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
  # Store the MRI and masks
  ax1.volume = mri
  ax1.mask = pred_mask
  ax1.is_contour = True
  ax1.volume_is_mask = False
  # Calc number of unique mask labels
  labels = list(np.unique(pred_mask))
  ax1.labels = labels  
  # Start showing mri with middle slice
  ax1.index = mri.shape[0] // 2
  ax1.imshow(mri[ax1.index], cmap=plt.cm.gray)
  add_contours(ax1)
  ax1.axis('off')
  ax1.set_title('Pred. Mask over MRI')
  
  ax2.volume = true_mask
  ax2.mask   = pred_mask
  ax2.is_contour = True
  ax2.volume_is_mask = True
  ax2.labels = labels
  ax2.index  = ax1.index
  ax2.imshow(true_mask[ax2.index], cmap=plt.cm.gray)
  add_contours(ax2)
  ax2.axis('off')
  ax2.set_title('Pred. Mask over True Mask')
  
  fig.canvas.mpl_connect('key_press_event', process_key)
  fig.suptitle(title)
  plt.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.3)
  global txt
  txt = plt.figtext(.6, .9, "Showing Slice # {:<0d}".format(ax1.index))
  #plt.tight_layout()
  plt.show()
  
def remove_keymap_conflicts(new_keys_set):
  for prop in plt.rcParams:
    if prop.startswith('keymap.'):
      keys = plt.rcParams[prop]
      remove_list = set(keys) & new_keys_set
      for key in remove_list:
        keys.remove(key)
  
def process_key(event):
  # Move forward/backward through slices for overlay viewer
  fig = event.canvas.figure
  ax1 = fig.axes[0]
  ax2 = fig.axes[1]
  if event.key == 'j':
    previous_slice(ax1)
    previous_slice(ax2)
  elif event.key == 'k':
    next_slice(ax1)
    next_slice(ax2)
  fig.canvas.draw()
  
def process_key_simple_viewer(event):
  # Move forward/backward through slices for simple 3D image viewer
  is_overlay = False
  fig = event.canvas.figure
  ax1 = fig.axes[0]
  if event.key == 'j':
    previous_slice(ax1, is_overlay)
  elif event.key == 'k':
    next_slice(ax1, is_overlay)
  fig.canvas.draw()
  
def previous_slice(ax, is_overlay=True):
  volume = ax.volume
  title = ax.get_title()

  ax.clear()
  ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
  #ax.images[0].set_array(volume[ax.index])
  ax.imshow(volume[ax.index], cmap=plt.cm.gray)
  if is_overlay is True:
    add_contours(ax)
  ax.axis('off')
  ax.set_title(title)
  global txt
  txt.remove()
  txt = plt.figtext(.6, .9, "Showing Slice # {:<0d}".format(ax.index))
  
def next_slice(ax, is_overlay=True):
  volume = ax.volume
  title = ax.get_title()
  
  ax.clear()
  ax.index = (ax.index + 1) % volume.shape[0]
  ax.clear()
  ax.imshow(volume[ax.index], cmap=plt.cm.gray)
  #ax.images[0].set_array(volume[ax.index])
  if is_overlay is True:
    add_contours(ax)
  ax.axis('off')
  ax.set_title(title)
  global txt
  txt.remove()
  txt = plt.figtext(.6, .9, "Showing Slice # {:<0d}".format(ax.index))
  
def add_contours(ax):
  mask = ax.mask[ax.index]
  labels = ax.labels
  colors = ['k', 'r', 'g', 'b', 'y']
  
  for idx, lbl in enumerate(labels):
    curr_mask_lbl = np.zeros_like(mask)
    curr_mask_lbl[np.where(mask == lbl)] = 1
    ax.contour(curr_mask_lbl, [0.5], linewidths=0.7, colors=colors[idx])
    