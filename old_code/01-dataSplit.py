# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 20:43:08 2023

@author: Andika eka
"""

import os
import random
import tensorflow as tf
import shutil 
# Define paths
dataset_path = './Original'
mask_path = "./Mask"
train_path = './train'  # Path to store train set
val_path = './validation'  # Path to store validation set
test_path = './test'  # Path to store test set

# Define train, validation, and test ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

if os.path.exists(train_path):
    shutil.rmtree(train_path)
    
if os.path.exists(val_path):
    shutil.rmtree(val_path)
    
if os.path.exists(test_path):
    shutil.rmtree(test_path)
    
# Create directories for train, validation, and test sets

def make_file_tree(path):
    os.makedirs(path)
    os.makedirs(path+'/images/')
    os.makedirs(path+ '/labels/')
    
make_file_tree(train_path)
make_file_tree(val_path)
make_file_tree(test_path)

# Get list of all files in the dataset directory
file_list = os.listdir(dataset_path)

# Shuffle the file list
random.shuffle(file_list)

# Split the file list into train, validation, and test sets
train_split = int(len(file_list) * train_ratio)
val_split = int(len(file_list) * val_ratio)

train_files = file_list[:train_split]
val_files = file_list[train_split:train_split+val_split]
test_files = file_list[train_split+val_split:]

# Copy files to their respective directories
for file in train_files:
    file_path = dataset_path+'/'+ file
    label_path = mask_path +'/'+ file.replace('O','M')
    shutil.copyfile(file_path, train_path +'/'+'/images/'  +'/'+ file)
    shutil.copyfile(label_path, train_path +'/'+ '/labels/' +'/'+ file)

for file in val_files:
    file_path = dataset_path+'/'+ file
    label_path = mask_path +'/'+ file.replace('O','M')
    shutil.copyfile(file_path, val_path +'/'+'/images/'  +'/'+ file)
    shutil.copyfile(label_path, val_path +'/'+ '/labels/' +'/'+ file)

for file in test_files:
    file_path = dataset_path+'/'+ file
    label_path = mask_path +'/'+ file.replace('O','M')
    shutil.copyfile(file_path, test_path +'/'+'/images/'  +'/'+ file)
    shutil.copyfile(label_path, test_path +'/'+ '/labels/' +'/'+ file)
