# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 19:33:08 2023

@author: Andika eka
"""
import shutil
import os
import cv2

# Set the input and output directories
image_path = './images'
mask_path = "./labels"
out_path = './resizeDATA'
out_image = out_path + '/images'
out_mask = out_path + '/labels'
 
if os.path.exists(out_path):
    shutil.rmtree(out_path)
    
os.mkdir(out_path)
os.mkdir(out_image)
os.mkdir(out_mask)
    

# Set the new size
new_size = (512, 512)

# Loop over all the files in the input directory
for filename in os.listdir(image_path):
    # Read the image
    img = cv2.imread(os.path.join(image_path, filename))
    
    # Resize the image
    resized_img = cv2.resize(img, new_size)
    
    # Save the resized image to the output directory
    cv2.imwrite(os.path.join(out_image, filename), resized_img)
    
# Loop over all the files in the input directory
for filename in os.listdir(mask_path):
    # Read the image
    img = cv2.imread(os.path.join(mask_path, filename))
    
    # Resize the image
    resized_img = cv2.resize(img, new_size)
    
    # Save the resized image to the output directory
    cv2.imwrite(os.path.join(out_mask, filename), resized_img)
    

out_path = './cropDATA'
out_image = out_path + '/images'
out_mask = out_path + '/labels'
 
if os.path.exists(out_path):
    shutil.rmtree(out_path)
    
os.mkdir(out_path)
os.mkdir(out_image)
os.mkdir(out_mask)

# Set the crop dimensions (x, y, w, h)
crop_dims = (128, 128, 512, 512)

# Loop over all the files in the input directory
for filename in os.listdir(image_path):
    # Read the image
    img = cv2.imread(os.path.join(image_path, filename))
    
    x, y, w, h = crop_dims
    cropped_img = img[y:y+h, x:x+w]
    
    # Save the resized image to the output directory
    cv2.imwrite(os.path.join(out_image, filename), cropped_img)
    
# Loop over all the files in the input directory
for filename in os.listdir(mask_path):
    # Read the image
    img = cv2.imread(os.path.join(mask_path, filename))
    
    x, y, w, h = crop_dims
    cropped_img = img[y:y+h, x:x+w]
    
    # Save the resized image to the output directory
    cv2.imwrite(os.path.join(out_mask, filename), cropped_img)
    
