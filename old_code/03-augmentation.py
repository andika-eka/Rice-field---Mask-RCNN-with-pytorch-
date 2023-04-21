# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 22:29:44 2023

@author: Andika eka
"""
import cv2
import numpy as np
import os

def augment_image(image, label):
    # Randomly flip image horizontally
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
    
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 0)

    # Randomly rotate image by 90 degrees
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    return image, label

image_path = './images'
mask_path = "./labels"


for file1, file2 in zip(os.listdir(image_path), os.listdir(mask_path)):
    img = cv2.imread(os.path.join(image_path, file1))
    label = cv2.imread(os.path.join(mask_path, file2))  
    augmented_img, augmented_label = augment_image(img,label)
    cv2.imwrite(os.path.join(image_path, "aug"+file1), augmented_img)
    cv2.imwrite(os.path.join(mask_path, "aug"+file2), augmented_label)
