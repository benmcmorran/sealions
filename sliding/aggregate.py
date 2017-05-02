import json
import pandas
import cv2
import numpy as np
from keras.models import load_model

"""
    Runs a model on sliced images and joins results
"""

MODEL_PATH = 'model_target_res_reg.h5'
INSTANCES = 'sliced_instances.json'
RESULTS = 'val_results.csv'

with open(INSTANCES) as in_f:
    instances = json.load(in_f)

with open('full_instances.json') as in_f:
    f_instances = json.load(in_f)

all_images = f_instances['images'] 
# images set aside for validation
full_to_test = set([ img['id'] for img in all_images[600:800]])
_images = instances['images']
images = []
for img in _images:
    key = int(img['id'].split('-')[0])
    if key in full_to_test:
	images.append(img)

annotations = instances['annotations']

model = load_model(MODEL_PATH)

test_results = {}

for img in images:
    parent = img['file_name'].split('-')[0]
    file_name = img['file_name']
    raw_img = cv2.imread('./data/SlicedTrain/{0}'.format(file_name))
    y = model.predict(np.array([raw_img]))
    if parent in test_results:
        test_results[parent] += y
    else:
        test_results[parent] = y
final_results = []
for img, pop in test_results.iteritems():
    record = [0, 0, 0, 0, 0, 0]
    record[0] = img
    record[1:] = pop[0]
    final_results.append(record)

df = pandas.DataFrame(final_results)
df.to_csv(RESULTS, header=False, index=False)
