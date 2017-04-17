import numpy as np
from scipy import misc # feel free to use another image loader
import glob
import pandas
import re
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