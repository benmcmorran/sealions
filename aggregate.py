import json
import pandas
import cv2
import numpy as np
from keras.models import load_model

"""
    Runs a model on sliced images and joins results
    NOTES: my machine is running low on memory, its possible
    I'll have to split my test images *live*, in which case
    this will have to operate differently test vs train.
"""

MODEL_PATH = 'model_target_full_res.h5'
INSTANCES = 'sliced_instances.json'
RESULTS = 'train_results.csv'

with open(INSTANCES) as in_f:
    instances = json.load(in_f)

images = instances['images']
annotations = instances['annotations']

model = load_model(MODEL_PATH)

test_results = {}

for img in images:
    parent = img['file_name'].split('-')[0]
    file_name = img['file_name']
    raw_img = cv2.imread('./data/SlicedTrain/{0}'.format(file_name))
    y = model.predict(np.array([raw_img]))
    #print parent
    if parent in test_results:
        test_results[parent] += y
    else:
        test_results[parent] = y
print test_results
final_results = []
for img, pop in test_results.iteritems():
    print img
    print pop
    record = [0, 0, 0, 0, 0, 0]
    record[0] = img
    record[1:] = pop[0]
    final_results.append(record)

df = pandas.DataFrame(final_results, header=False, index=False)
df.to_csv(RESULTS)
