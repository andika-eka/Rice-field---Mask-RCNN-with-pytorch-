
"""
Created on Mon Feb  6 01:18:42 2023

@author: putua
"""
import shutil
import os
import imageio
import numpy as np

mask_path = "./labels"
image_path = './images'
out_path = './intDATA'
ext = '.png'
out_image = out_path +"/image"
out_mask = out_path +"/mask"

if os.path.exists(out_path):
    shutil.rmtree(out_path)
    
os.mkdir(out_path)
os.mkdir(out_image)
os.mkdir(out_mask)


for filename in os.listdir(image_path):
    if filename.endswith(ext):
       image = imageio.imread(image_path+'/'+filename)
       out_file = out_path+"/image"+'/'+filename.replace(ext, '')+'.npy'
       np.save(out_file, image)
       
for filename in os.listdir(mask_path):
    if filename.endswith(ext):
        binary_mask = imageio.imread(mask_path+'/'+filename)
        integer_mask = np.where(binary_mask != 0, 1, 0).astype(np.uint8)
        out_file = out_path+'/mask'+'/'+filename.replace(ext, '')+'.npy'
        np.save(out_file, integer_mask)

        
       
        