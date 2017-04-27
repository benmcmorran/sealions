"""Catalogs all the bounding boxes to json.

working off of: https://www.kaggle.com/ranbato/noaa-fisheries-steller-sea-lion-population-count/finding-the-dots

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

# Filter ranges for each dot color.
redRange = [np.array([160, 0, 0]), np.array([255, 50, 50])]
magnetaRange = [np.array([160, 0, 160]), np.array([255, 50, 255])]
brownRange = [np.array([76, 39, 5]), np.array([94, 53, 22])]
blueRange = [np.array([0, 0, 160]), np.array([56, 56, 255])]
greenRange = [np.array([0, 160, 0]), np.array([56, 255, 56])]
colorRanges = [redRange, magnetaRange, brownRange, blueRange, greenRange ]

red = (255, 0, 0)
magneta = (255, 0, 255)
brown = (78, 42, 8)
blue = (0, 0, 255)
green = (0, 255, 0)
colors = [red, magneta, brown, blue, green]

info = {}
images = []
annotations = []
annotation_id = 0
img_names = glob.glob("./data/TrainDotted/*.jpg")

for i, img_name in enumerate(img_names):
    # tick
    print 'processing image: {0}'.format(i)

    # load image
    img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)

    # create json image description
    image_json = {
        'id' : int(img_name.replace('./data/TrainDotted/','').replace('.jpg','')),
        'width' : np.shape(img)[1],  # width is 1, not 0!!
        'height' : np.shape(img)[0], # originally had this wrong
        'file_name' : img_name.replace('./data/TrainDotted/','')
    }
    images.append(image_json)

    # find bounding boxes
    counts = np.zeros(5)
    for color in range(0,5):
        cmsk = cv2.inRange(img, colorRanges[color][0], colorRanges[color][1])
        circles = cv2.HoughCircles(cmsk,cv2.cv.CV_HOUGH_GRADIENT,1,50, param1=40,param2=1,minRadius=2,maxRadius=10)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            counts[color] = len(circles[0,:])

            for i in circles[0,:]:
                # create json annotation
                # this is mediumly good, -25 + 25...
                annotations.append({
                    "id" : annotation_id,
                    "image_id" : image_json['id'],
                    "category_id" : color,
                    # TODO: confirm this order is correct! 0,1 for x,y
                    "bbox" : [i[0].item(), i[1].item(), 50, 50]
                })
                annotation_id += 1
                print 'creating annotation: {0}'.format(annotation_id)

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