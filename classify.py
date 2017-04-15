from keras.models import Sequential
from keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras import backend as K

# Based on http://stackoverflow.com/questions/41458859/keras-custom-metric-for-single-class-accuracy

def single_class_metric(class_id, metric):
    def fn(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_pred = K.argmax(y_pred, axis=-1)

        if metric == 'precision':
            accuracy_mask = K.cast(K.equal(class_id_pred, class_id), 'int32')
        elif metric == 'recall':
            accuracy_mask = K.cast(K.equal(class_id_true, class_id), 'int32')
        
        correct_pred = K.cast(
            K.equal(class_id_pred, class_id_true),
        'int32') * accuracy_mask
        return K.sum(correct_pred) / K.maximum(K.sum(accuracy_mask), 1)
    fn.__name__ = 'pre' if metric == 'precision' else 'rec'
    return fn

# Based on https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

batch_size = 16
datagen = ImageDataGenerator(
    rescale=1/255
)

# default padding is valid, not same
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=[
        'accuracy',
        single_class_metric(1, 'precision'),
        single_class_metric(1, 'recall')
    ]
)

train_generator = datagen.flow_from_directory(
    'Thumbnails/Train',
    target_size=(32, 32),
    batch_size=batch_size,
    classes=['LION', 'EMPTY']
)

validation_generator = datagen.flow_from_directory(
    'Thumbnails/Validation',
    target_size=(32, 32),
    batch_size=batch_size,
    classes=['LION', 'EMPTY']
)

model.fit_generator(
    train_generator,
    steps_per_epoch=77000 // batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=22000 // batch_size,
    callbacks=[
        ModelCheckpoint('weights4.{epoch:02d}-{val_loss:.2f}.hdf5'),
        CSVLogger('training4.csv')
    ],
    class_weight={
        0: 10,
        1: 1
    }
)

# import cv2
# import numpy as np
# import glob
# import os
# import time

# start_time = time.time()
# with open('counts2.csv', 'w', 1) as counts:
#     counts.write('train_id,pct_covered\n')

#     filenames = glob.glob('Full/Train/*.jpg')
#     for i, filename in enumerate(filenames):
#         train_id = os.path.basename(filename).split('.')[0]
#         image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
#         features = np.zeros((image.shape[0] // 32 - 1, image.shape[1] // 32 - 1))

#         for y in range(0, image.shape[0] - 64 + 1, 32):
#             pct_complete = (i + y / image.shape[0]) / len(filenames)
#             now = time.time()
#             eta = int((now - start_time) * (1 / pct_complete - 1)) \
#                 if pct_complete != 0 else '---'
#             print('{:.2%}\t({:.2%} on {})\tETA {}s     '.format(
#                 pct_complete,
#                 y / image.shape[0],
#                 filename,
#                 eta
#             ), end='\r')

#             coords = []
#             thumbnails = []
#             for x in range(0, image.shape[1] - 64 + 1, 32):
#                 thumbnails.append(cv2.resize(image[y:y+64, x:x+64], (32, 32)))
#                 coords.append((y // 32, x // 32))

#             thumbnails = np.stack(thumbnails)
#             predictions = model.predict(thumbnails)
#             for j in range(len(coords)):
#                 features[coords[j][0], coords[j][1]] = predictions[j, 0] / predictions[j].sum()

#         counts.write(train_id + ',' + str(features.sum() / features.size) + '\n')
#         features = features * 255
#         cv2.imwrite(os.path.join('Full', 'FeatureMaps2', train_id + '.png'), features)

#     print()