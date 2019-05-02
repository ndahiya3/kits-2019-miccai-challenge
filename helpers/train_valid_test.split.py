#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:52:52 2019
    Split MRI datasets into train/valid/test (50,20,30)
    Testing set will not be touched except to report final results
    for different models. Valid set will be used for parameter tuning
    model selection etc.
    Available dataset IDs are in 'case_names.txt'
@author: ndahiya, aloksh
"""

import numpy as np
from sklearn.model_selection import train_test_split

found_ids = []
with open('case_names.txt', 'r') as f:
    for line in f:
        found_ids.append(line.strip())
        
print("Total datasets: {}".format(len(found_ids)))

# Total datasets 701. Train: 50% 350, Valid: 14% 100, Test: 36% 251
# First split 60/40 to get train set 
ids_train, ids_rest = train_test_split(np.asarray(found_ids), train_size=0.5, random_state=1)
        
# Now split rest as 20/30 to get valid and test sets
ids_valid, ids_test = train_test_split(ids_rest, train_size=0.3, random_state=2)

print('Images in train: ', ids_train.size)
print('Images in valid: ', ids_valid.size)
print('Images in test : ', ids_test.size)

# Write train/valid/test files
with open('train_ids.txt', 'w') as f:
    for ids in list(ids_train):
        f.write('%s\n' % ids)
with open('valid_ids.txt', 'w') as f:
    for ids in list(ids_valid):
        f.write('%s\n' % ids)
with open('test_ids.txt', 'w') as f:
    for ids in list(ids_test):
        f.write('%s\n' % ids)
        
