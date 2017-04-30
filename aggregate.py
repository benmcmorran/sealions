import json
import pandas

"""
    Runs a model on sliced images and joins results
    NOTES: my machine is running low on memory, its possible
    I'll have to split my test images *live*, in which case
    this will have to operate differently test vs train.
"""

MODEL_PATH = 'model.h5'
INSTANCES = 'test_instances.json'
RESULTS = 'test_results.csv'

with open(INSTANCES) as in_f:
    instances = json.load(in_f)

images = instances['images']
annotations = instances['annotations']

model = load_model(MODEL_PATH)

test_results = {}

for img in images:
    parent = img['original_id']
    file_name = img['file_name']
    raw_img = cv2.imread('./data/Sliced/{0}'.format(file_name))
    y = model.predict(np.array([raw_img]))
    if parent in test_results:
        test_results[parent] += y
    else:
        test_results[parent] = y

final_results = []
for img, pop in test_results.iteritems():
    record = [0, 0, 0, 0, 0, 0]
    record[0] = img
    record[1:] = pop
    final_results.append(record)

df = pandas.DataFrame(final_results)
df.to_csv(RESULTS)