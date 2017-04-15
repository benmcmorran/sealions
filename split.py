import os
import random

VALIDATION_RATIO = .2
TEST_RATIO = .1
SOURCE_DIR = 'Extracted'
DEST_DIR = 'Thumbnails'

lion_types = os.listdir(SOURCE_DIR)

for lion_type in lion_types:
    base = os.path.join(SOURCE_DIR, lion_type)
    images = os.listdir(base)
    random.shuffle(images)

    print('Processing {} images of type {}'.format(len(images), lion_type))
    for i, image in enumerate(images):
        if i < len(images) * VALIDATION_RATIO:
            dir_name = 'Validation'
        elif i < len(images) * (VALIDATION_RATIO + TEST_RATIO):
            dir_name = 'Test'
        else:
            dir_name = 'Train'
        os.renames(
            os.path.join(base, image),
            os.path.join(DEST_DIR, dir_name, lion_type, image)
        )