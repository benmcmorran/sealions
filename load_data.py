import numpy as np
import cv2
import glob
import pandas
import re
import json
df = pandas.read_csv('./data/Train/train.csv')
list_of_images = glob.glob('./data/Train/*.jpg')
raw_values = df.values[:,1:]

def train_generator(batch_size=5):
    """creates batches of images and labels

    Images are force to smaller of two sizes
    Labels are the 5 class counts
    """
    images = []
    labels = []
    for i,img in enumerate(list_of_images):
        raw_img = misc.imread(img)
        # forcing larger images to smaller size
        resize_img = misc.imresize(raw_img, (4992, 3328, 3))
        images.append(resize_img)
        file_name = re.sub("[^0-9]", "", img)
        ind = int(file_name)
        label = raw_values[ind,]
        labels.append(label)
    images = np.asarray(images)
    labels = np.asarray(labels)
    total = len(list_of_images)
    while (True):
        for i in range(0,total,batch_size):
            yield(images[i:i+batch_size],labels[i:i+batch_size])

def train_sliced_generator(batch_size=5):
    """creates batches of images and labels

    Images are force to smaller of two sizes
    Labels are the 5 class counts
    Converts annotations to labels
    """
    with open('sliced_small_instances.json') as in_f:
        instances = json.load(in_f)
    images = instances['images']
    annotations = instances['annotations']
    j = 1
    batch_images = []
    batch_labels = []
    for i,img in enumerate(images):
        file_name = img['file_name']
        raw_img = cv2.imread('./data/Sliced/{0}'.format(file_name))
        label = np.array([0,0,0,0,0])
        correct_ann = [ann for ann in annotations if ann['image_id'] == file_name]
        for ann in correct_ann:
            print ann
            label[ann['category_id'] - 1] += 1
                
        batch_images.append(raw_img)
        batch_labels.append(label)
        if j == batch_size:
            yield(np.array(batch_images), np.array(batch_labels))
            batch_images = []
            batch_labels = []
            j = 1
        else:
            j += 1

def train_sliced(batch_size=5):
    """creates batches of images and labels

    Images are force to smaller of two sizes
    Labels are the 5 class counts
    Converts annotations to labels
    """
    with open('sliced_small_instances.json') as in_f:
        instances = json.load(in_f)
    images = instances['images']
    annotations = instances['annotations']
    j = 1
    batch_images = []
    batch_labels = []
    for i,img in enumerate(images):
        file_name = img['file_name']
        raw_img = cv2.imread('./data/Sliced/{0}'.format(file_name))
                
        label = np.array([0,0,0,0,0])
        correct_ann = [ann for ann in annotations if ann['image_id'] == file_name]
        for ann in correct_ann:
            label[ann['category_id'] - 1] += 1
        batch_images.append(raw_img)
        batch_labels.append(label)
        if i > batch_size:
            break
    return np.array(batch_images), np.array(batch_labels)

