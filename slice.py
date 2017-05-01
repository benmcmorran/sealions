"""
This file:
    1. slices images into smaller images
    2. creates updated instances json file
"""

FULL_INSTANCES = 'instances.json'
SLICED_INSTANCES = 'sliced_instances.json'
TARGET_SLICED_INSTANCES = 'target_sliced_instances.json'

import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json

new_h = 416
new_w = 624
data_dir = './data/Train/{0}'
slice_dir = './data/SlicedTrain/{0}'

with open(FULL_INSTANCES) as in_f:
    instances = json.load(in_f)
    images = instances['images']
    annotations = instances['annotations']

    sliced_annotations = []
    target_sliced_annotations = []
    sliced_images = []
    target_sliced_images = []

    annotation_id = 0

    for img in images:
        img_id = img['id']
        w = img['width']
        h = img['height']
        file_name = img['file_name']

        loaded_img = cv2.imread('./data/Train/{0}'.format(file_name))
        img_annotations = [ a for a in annotations if a['image_id'] == img_id]


        slices_width = w / new_w
        slices_height = h / new_h

        for i in xrange(slices_width):
            for j in xrange(slices_height):
                # start in top left
                crop_name = '{0}-{1}-{2}.jpg'.format(img_id, i, j)
                crop = loaded_img[ j*new_h:(j*new_h+new_h), i*new_w:(i*new_w+new_w) ]
                
                cv2.imwrite('./data/SlicedTrain/{0}'.format(crop_name), crop)

                # save image information
                image_json = {
                    'id' : crop_name.replace('.img',''),
                    'width' : new_w,
                    'height' : new_h,
                    'file_name' : crop_name
                }

                # find annotations
                found = 0
                for ann in img_annotations:
                    # check if annotation is in new image
                    # original annotations were cornered on label
                    x,y = ann['bbox'][0], ann['bbox'][1]
                    if  j*new_h < y and y < (j*new_h+new_h) and i*new_w < x and x < (i*new_w+new_w):
                        ann = {
                            "id" : annotation_id,
                            "image_id" : crop_name,
                            "original_id" : img_id,
                            "category_id" : ann['category_id'],
                            "bbox" : [x - 25 - i*new_w, y - 25 - j*new_h, 50, 50]
                        }
                        target_sliced_annotations.append(ann)
                        sliced_annotations.append(ann)
                        found += 1
                        annotation_id += 1
                if found > 0:
                    target_sliced_images.append(image_json)
                sliced_images.append(image_json)
    
    sliced_instances = {
        'images' : sliced_images,
        'annotations' : sliced_annotations
    }

    target_sliced_instances = {
        'images' : target_sliced_images,
        'annotations' : target_sliced_annotations
    }

    with open(SLICED_INSTANCES, "w") as out:
        json.dump(sliced_instances, out, indent=1)
    with open(TARGET_SLICED_INSTANCES, "w") as out:
        json.dump(target_sliced_instances, out, indent=1)
