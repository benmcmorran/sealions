import os
import glob
import math
from enum import Enum, unique
import random
import cv2
import skimage.feature
from matplotlib import pyplot as plt

# Code adapted from https://www.kaggle.com/radustoicescu/noaa-fisheries-steller-sea-lion-population-count/use-keras-to-classify-sea-lions-0-91-accuracy

@unique
class SeaLionType(Enum):
    ADULT_MALE = 1
    SUBADULT_MALE = 2
    PUP = 3
    JUVENILE = 4
    ADULT_FEMALE = 5

    @classmethod
    def from_color(cls, r, g, b):
        if r > 200 and g < 50 and b < 50: # RED
            return cls.ADULT_MALE
        elif r > 200 and g > 200 and b < 50: # MAGENTA
            return cls.SUBADULT_MALE
        elif r < 100 and g < 100 and 150 < b < 200: # GREEN
            return cls.PUP
        elif r < 100 and  100 < g and b < 100: # BLUE
            return cls.JUVENILE
        elif r < 150 and g < 50 and b < 100:  # BROWN
            return cls.ADULT_FEMALE
        else:
            return None

def extract_thumbnail(img, x, y, cls, name):
    y = int(y)
    x = int(x)
    thumbnail = img[y - 32 : y + 32, x - 32 : x + 32, :]
    if thumbnail.shape != (64, 64, 3):
        return False
    
    path = os.path.join('D:/KaggleNOAASeaLions/Extracted', cls, str(name) + '.jpg')
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    cv2.imwrite(path, thumbnail)

    return True

def is_location_empty(y, x, blobs):
    for by, bx, _ in blobs:
        dy = y - by
        dx = x - bx
        if math.sqrt(dy * dy + dx * dx) < 100:
            return False
    return True

def empty_location(shape, blobs):
    while True:
        y, x = (random.randrange(shape[0]), random.randrange(shape[1]))
        if is_location_empty(y, x, blobs):
            return (y, x)

lion_counts = { lion_type: 0 for lion_type in SeaLionType }
empty_count = 0

file_names = (
    os.path.basename(path) for path 
    in glob.iglob('D:/KaggleNOAASeaLions/Train/*.jpg')
)

mismatched = [3,7,9,21,30,34,71,81,89,97,151,184,215,234,242,268,290,311,331,344,380,384,406,421,469,475,490,499,507,530,531,605,607,614,621,638,644,687,712,721,767,779,781,794,800,811,839,840,869,882,901,903,905,909,913,927,946]

for name in file_names:
    print('Processing {}'.format(name))

    if int(name.split('.')[0]) in mismatched:
        print('Skipping mismatched image')
        continue 
    
    prefix = 'D:/KaggleNOAASeaLions'
    real = cv2.imread(os.path.join(prefix, 'Train', name))
    dotted = cv2.imread(os.path.join(prefix, 'TrainDotted', name))

    if real is None or dotted is None:
        print('Skipping unpaired image')
        continue

    diff = cv2.absdiff(real, dotted)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    mask = cv2.cvtColor(dotted, cv2.COLOR_BGR2GRAY)
    mask[mask < 20] = 0
    mask[mask > 0] = 255
    diff = cv2.bitwise_or(diff, diff, mask=mask)

    blobs = skimage.feature.blob_log(diff, min_sigma=3, max_sigma=4, num_sigma=1, threshold=0.02)

    for y, x, _ in blobs:
        g, b, r = dotted[int(y)][int(x)][:]
        lion_type = SeaLionType.from_color(r, g, b)
        if not lion_type:
            continue

        if extract_thumbnail(real, x, y, lion_type.name, lion_counts[lion_type]):
            lion_counts[lion_type] += 1

    for y, x in (empty_location(real.shape, blobs) for _ in range(50)):
        if extract_thumbnail(real, x, y, 'EMPTY', empty_count):
            empty_count += 1
