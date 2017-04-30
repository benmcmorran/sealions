"""Catalogs all the bounding boxes to json.

Writes annotations to 'instances.json'

Save dots to a file as output for object detection
Following COCO json formatting http://mscoco.org/dataset/#download

NOTE IMPORTANT: the bounding boxes top left corner is (should be) the dot,
                and then it extends down and to the right by 50. This was
                a mistake, but is currently kept as this helps the slice
                function - and really the bounding boxes aren't even great
                ground truth labels. 
"""
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import skimage.feature
import gc
gc.enable()


mismatched = set([3,7,9,21,30,34,71,81,89,97,151,184,215,234,242,268,290,311,331,344,380,384,406,421,469,475,490,499,507,530,531,605,607,614,621,638,644,687,712,721,767,779,781,794,800,811,839,840,869,882,901,903,905,909,913,927,946])

# Filter ranges for each dot color.
info = {}
images = []
annotations = []
annotation_id = 0
img_names = glob.glob("./data/Train/*.jpg")
img_names = [i.replace('./data/Train/','') for i in img_names]

for i, img_name in enumerate(img_names):
    # skip mis-matched 
    if int(img_name.replace('.jpg','')) in mismatched:
	continue    

    # tick
    print 'processing image: {0}'.format(i)

    # read the Train and Train Dotted images
    image_1 = cv2.imread("./data/TrainDotted/" + img_name)
    image_2 = cv2.imread("./data/Train/" + img_name)
    
    # absolute difference between Train and Train Dotted
    image_3 = cv2.absdiff(image_1,image_2)

    print np.shape(image_1)
    print img_name
    # create json image description
    image_json = {
        'id' : int(img_name.replace('./data/TrainDotted/','').replace('.jpg','')),
        'width' : np.shape(image_1)[1],  # width is 1, not 0!!
        'height' : np.shape(image_1)[0], # originally had this wrong
        'file_name' : img_name.replace('./data/TrainDotted/','')
    }
    images.append(image_json)

    # mask out blackened regions from Train Dotted
    mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    mask_1[mask_1 < 20] = 0
    mask_1[mask_1 > 0] = 255
    
    mask_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    mask_2[mask_2 < 20] = 0
    mask_2[mask_2 > 0] = 255
    
    image_4 = cv2.bitwise_or(image_3, image_3, mask=mask_1)
    image_5 = cv2.bitwise_or(image_4, image_4, mask=mask_2) 
    
    # convert to grayscale to be accepted by skimage.feature.blob_log
    image_6 = cv2.cvtColor(image_5, cv2.COLOR_BGR2GRAY)
    
    # detect blobs
    blobs = skimage.feature.blob_log(image_6, min_sigma=3, max_sigma=4, num_sigma=1, threshold=0.02)
    
    # prepare the image to plot the results on
    image_7 = cv2.cvtColor(image_6, cv2.COLOR_GRAY2BGR)
    
    for blob in blobs:
        # get the coordinates for each blob
        y, x, s = blob
        # get the color of the pixel from Train Dotted in the center of the blob
        b,g,r = image_1[int(y)][int(x)][:]
        
        seal_type = 0

        # decision tree to pick the class of the blob by looking at the color in Train Dotted
        if r > 200 and b < 50 and g < 50: # RED
            seal_type = 1         
        elif r > 200 and b > 200 and g < 50: # MAGENTA
            seal_type = 2     
        elif r < 100 and b < 100 and 150 < g < 200: # GREEN
            seal_type = 3
        elif r < 100 and  100 < b and g < 100: # BLUE
            seal_type = 4
        elif r < 150 and b < 50 and g < 100:  # BROWN
            seal_type = 5
        else:
            print 'continuing'
            continue

        annotations.append({
            "id" : annotation_id,
            "image_id" : image_json['id'],
            "category_id" : seal_type,
            "bbox" : [x, y, 50, 50]
        })

        annotation_id += 1

info['images'] = images
info['annotations'] = annotations

import json
with open("instances.json", "w") as out:
    json.dump(info, out, indent=2)

"""
## load instances
In [1]: with open('instances.json') as in_f:
   ...:     data = json.load(in_f)
"""
