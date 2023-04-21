# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 01:18:42 2023

@author: putua
"""
import os
import tensorflow as tf
import numpy as np


img_path = './intDATA/image'
mask_path = './intDATA/mask'
ext = '.npy'

image_data = tf.data.Dataset.from_tensor_slices([])
first = True
for filename in os.listdir(img_path):
    if filename.endswith(ext):
        images = np.load(img_path+'/'+filename)
        images = tf.convert_to_tensor(images, dtype=tf.float32)
        images = tf.data.Dataset.from_tensor_slices(images)
        print(filename)
        if first:
            image_data = images
            first = False
        else:
            image_data = image_data.concatenate(images)

mask_data = tf.data.Dataset.from_tensor_slices([])
first = True
for filename in os.listdir(mask_path):
    if filename.endswith(ext):
        masks = np.load(mask_path+'/'+filename)
        masks = tf.convert_to_tensor(masks, dtype=tf.float32)
        masks = tf.data.Dataset.from_tensor_slices(masks)
        print(filename)
        if first:
            mask_data = masks
            first = False
        else:
            mask_data = mask_data.concatenate(masks)


dataset = tf.data.Dataset.zip((image_data, mask_data))


dataset.save( "./dataset")
