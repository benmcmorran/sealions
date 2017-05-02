import numpy as np
import cv2
import glob
import pandas
import re
import json
from keras.utils import to_categorical

def multi_generator(batch_size=6):
    """generates batches of detection, regression, and total
    """

    with open('sliced_instances.json') as in_f:
        all_instances = json.load(in_f)

    all_images = all_instances['images']
    all_annotations = all_instances['annotations']

    label_map = {}
    for ann in all_annotations:
	if ann['image_id'] not in label_map:
	    label_map[ann['image_id']] = np.array([0.,0.,0.,0.,0.])
	label_map[ann['image_id']][ann['category_id'] - 1] += 1
    
    annotation_map = {}
    for img in all_images:
        file_name = img['file_name']
        if file_name in label_map:
	    label = label_map[file_name]
	else:
	    label = np.array([0.,0.,0.,0.,0.])
        if np.sum(label) > 0:
            detect = np.array([1., 0]) # to_categorical(1.)
        else:
            detect = np.array([0., 1]) # to_categorical(0.)
        annotation_map[file_name] = (label, detect)
    N = len(all_images)
    while True:
        i = 0
        while i + batch_size < N:
            
            batch_images = []
            batch_labels = []
            batch_detect = []

            for _i in xrange(batch_size):
                i += 1
    		j = np.random.choice(N, 1)
		img = all_images[j[0]]
                file_name = img['file_name']
                raw_img = cv2.imread('./data/SlicedTrain/{0}'.format(file_name))
                label, detect = annotation_map[file_name]
                batch_images.append(raw_img)
                batch_labels.append(label)
                batch_detect.append(detect)
            yield({'main_input' : np.array(batch_images)}, {'detect_output': np.array(batch_detect), 'regress_output': np.array(batch_labels), 'final_output': np.array(batch_labels)})

