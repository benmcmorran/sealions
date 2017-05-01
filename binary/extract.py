import os
import sys
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

def extract_thumbnail(img, x, y, cls, imnum, name):
    y = int(y)
    x = int(x)
    thumbnail = img[y - 64 : y + 64, x - 64 : x + 64, :]
    if thumbnail.shape != (128, 128, 3):
        return False
    
    path = os.path.join('D:/KaggleNOAASeaLions/Extracted', cls, str(imnum) + '_' + str(name) + '.jpg')
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

if len(sys.argv) >= 2 and sys.argv[1]:
    file_names = ['D:/KaggleNOAASeaLions/Train/' + str(sys.argv[1]) + '.jpg']
else:
    file_names = list(
        os.path.basename(path) for path 
        in glob.iglob('D:/KaggleNOAASeaLions/Train/*.jpg')
    )

mismatched = [3,7,9,21,30,34,71,81,89,97,151,184,215,234,242,268,290,311,331,344,380,384,406,421,469,475,490,499,507,530,531,605,607,614,621,638,644,687,712,721,767,779,781,794,800,811,839,840,869,882,901,903,905,909,913,927,946]

for i, name in enumerate(file_names):
    pct_complete = i / len(file_names)
    print('Processing {} ({:.2%} complete)'.format(name, pct_complete), end='\r')
    num = int(name.split('.')[0])

    if num in mismatched:
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
    
    mask1 = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY)
    mask1[mask1 < 20] = 0
    mask1[mask1 > 0] = 255

    mask2 = cv2.cvtColor(dotted, cv2.COLOR_BGR2GRAY)
    mask2[mask2 < 20] = 0
    mask2[mask2 > 0] = 255

    diff = cv2.bitwise_or(diff, diff, mask=mask1)
    diff = cv2.bitwise_or(diff, diff, mask=mask2)

    blobs = skimage.feature.blob_log(diff, min_sigma=3, max_sigma=4, num_sigma=1, threshold=0.02)
    annotated = real.copy()

    lion_count = 0
    for y, x, _ in blobs:
        g, b, r = dotted[int(y)][int(x)][:]
        lion_type = SeaLionType.from_color(r, g, b)
        if not lion_type:
            continue
        
        if extract_thumbnail(real, x, y, lion_type.name, num, lion_count):
            cv2.rectangle(annotated, (int(x) - 64, int(y) - 64), (int(x) + 64, int(y) + 64), (0, 0, 255), 2)
            cv2.putText(annotated, str(lion_count), (int(x) - 10, int(y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            lion_count += 1

    cv2.imwrite(os.path.join('D:/KaggleNOAASeaLions/Extracted', str(num) + '.jpg'), annotated)
    
    empty_count = 0
    for y, x in (empty_location(real.shape, blobs) for _ in range(75)):
        if extract_thumbnail(real, x, y, 'EMPTY', num, empty_count):
            empty_count += 1
