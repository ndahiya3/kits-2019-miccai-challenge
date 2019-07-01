#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 22:14:31 2019
Test .npy image/mask overlay
@author: ndahiya
"""

import numpy as np
import matplotlib.pyplot as plt
gray_path = 'unet_tversky_mini/train/dicoms/'
mask_path = 'unet_tversky_mini/train/masks/'

case_name = 'case_00036.39.npz'

gray_img = np.load(gray_path + case_name)['A']
mask_img = np.load(mask_path + case_name)['A']

print(gray_img.shape)
print(mask_img.shape)

fig=plt.figure(figsize=(8, 8))

fig.add_subplot(1, 2, 1)

plt.imshow(gray_img)
fig.add_subplot(1, 2, 2)
plt.imshow(mask_img)
plt.show()