import cv2
import os
import numpy as np
import torch
import random
import json

def crop_data(image, mask, dimension):
    
    x, y, w, h = dimension
    
    image = image[y:y+h, x:x+w]
    mask = mask[y:y+h, x:x+w]
    
    return image, mask

def resize_data(image, mask, size):
    
    image = cv2.resize(image, size)
    mask = cv2.resize(mask, size)
    
    return image, mask

def augment_data(image, mask):
    
    flip_image = [image]
    flip_mask  = [mask]
    
    tmp_image = cv2.flip(image, 1)
    tmp_mask = cv2.flip(mask, 1)
    
    flip_image.append(tmp_image)
    flip_mask.append(tmp_mask)
    
    tmp_image = cv2.flip(image, 0)
    tmp_mask = cv2.flip(mask, 0)
    
    flip_image.append(tmp_image)
    flip_mask.append(tmp_mask)
    
    augmented_image = flip_image.copy()
    augmented_mask = flip_mask.copy()
    rotate_code = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]
    
    for i in range(3):
        for image, mask in zip(flip_image, flip_mask):
            tmp_image = cv2.rotate(image, rotate_code[i])
            tmp_mask = cv2.rotate(mask, rotate_code[i])
            
            augmented_image.append(tmp_image)
            augmented_mask.append(tmp_mask)
    
    return augmented_image, augmented_mask

preprocessing_preset = [
    {
        "name" : "crop_512",
        "size" : (512, 512),
        "method" : "crop"
    },
    {
        "name" : "resize_512",
        "size" : (512, 512),
        "method" : "resize"
    },
    {
        "name" : "crop_128",
        "size" : (128, 128),
        "method" : "crop"
    },
    {
        "name" : "resize_128",
        "size" : (128, 128),
        "method" : "resize"
    },
]

import json
import shutil

def preprocessing(out_path, data_path, output_size, method, crop_coordiate = None):
    image_path_out = os.path.join(out_path,"image")
    mask_path_out = os.path.join(out_path,"mask")
    result_path_out = os.path.join(out_path, "result")

    if os.path.exists(image_path_out):
        shutil.rmtree(image_path_out)
    if os.path.exists(mask_path_out):
        shutil.rmtree(mask_path_out)
    if os.path.exists(result_path_out):
        shutil.rmtree(result_path_out)
    os.mkdir(result_path_out)
    os.mkdir(image_path_out)
    os.mkdir(mask_path_out)
    data_count = 0
    for data in data_path:
        image = cv2.imread(data["image"])
        mask = cv2.imread(data["label"])
        
        if method == "crop":
            image_crop, mask_crop = crop_data(image, mask, crop_coordiate)
            image_resize, mask_resize = resize_data(image_crop, mask_crop, output_size)
        elif method == "resize":
            image_resize, mask_resize = resize_data(image, mask, output_size)
            
        
        image_augmented, mask_augmented = augment_data(image_resize, mask_resize)
        
        for new_image, new_mask in zip(image_augmented, mask_augmented):
            cv2.imwrite(image_path_out + "/" + str(data_count) + ".png", new_image)
            cv2.imwrite(mask_path_out + "/" + str(data_count) + ".png", new_mask)
            data_count += 1

def convert2annotation(path, visualize=False):
    json_path = os.path.join(path, "json")
    out_image_path = os.path.join(path, "annotated")

    if os.path.exists(json_path):
        shutil.rmtree(json_path)
    if os.path.exists(out_image_path):
        shutil.rmtree(out_image_path)
    os.mkdir(json_path)
    os.mkdir(out_image_path)

    image_train_dir = os.path.join(path, "image")
    mask_train_dir = os.path.join(path, "mask")
    for mask_path, image_path in zip(os.listdir(mask_train_dir), os.listdir(image_train_dir)):
        mask = cv2 .imread(os.path.join(mask_train_dir, mask_path))
        image = cv2.imread(os.path.join(image_train_dir, image_path))    
        copy_image = image.copy()
        mask = cv2.bitwise_not(mask)
        gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        boxes = torch.zeros([len(contours),4], dtype=torch.float32)

        data = {
            "file" : image_path,
            "n_object": len(contours),
            "annotation" : [],
            "box" : []
        }    
        
        for i,contour in enumerate(contours) :
            x,y,w,h = cv2.boundingRect(contour)
            boxes[i] = torch.tensor([x,y, x+w, y+h])
            boxed_image = cv2.rectangle(copy_image, (x,y), (x+w, y+h), (0,255,255), 2)
            data["annotation"].append(contour.tolist())
            data["box"].append([x,y,x+w, y+h])
        with open(os.path.join(json_path,image_path.replace(".png", ".json")), "w") as file:
            json.dump(data,file)
            
        if visualize:
            cv2.drawContours(boxed_image, contours, -1, (0, 255, 0), 3)
            cv2.fillPoly(image, contours, (0, 255, 0, 100))
            image = cv2.addWeighted(image, 0.5, copy_image, 0.5, 0)
            cv2.imwrite(os.path.join(out_image_path,image_path), image)

data_train = json.loads(open('data_train.json').read())
data_validation = json.loads(open('data_validation.json').read())
data_test = json.loads(open('data_test.json').read())

for experiment in preprocessing_preset:
    if os.path.exists(experiment["name"]):
        shutil.rmtree(experiment["name"])

    os.mkdir(experiment["name"])
    
    preprocessing(experiment["name"], data_train, experiment["size"], experiment["method"], (500, 0 , 1080, 1080))
for experiment in preprocessing_preset:
    convert2annotation(experiment["name"], visualize=True)