# Used for binary thumbnail classification

from keras.models import Sequential
from keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
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
train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=360,
    width_shift_range=.05,
    height_shift_range=.05,
    zoom_range=.03,
    horizontal_flip=True,
    vertical_flip=True
)
validation_datagen = ImageDataGenerator(
    rescale=1/255
)

# default padding is valid, not same
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(lr=0.0001),
    metrics=[
        'accuracy'
    ]
)

# train_generator = train_datagen.flow_from_directory(
#     'Thumbnails2/Train',
#     target_size=(64, 64),
#     batch_size=batch_size,
#     classes=['LION', 'EMPTY']
# )

validation_generator = validation_datagen.flow_from_directory(
    'Thumbnails2/Validation',
    target_size=(64, 64),
    batch_size=batch_size,
    classes=['LION', 'EMPTY'],
    shuffle=False
)

model.load_weights('data/augmented_thumb2/weights_lr-4.49-0.04.hdf5')

# model.fit_generator(
#     train_generator,
#     steps_per_epoch=88284 // batch_size,
#     epochs=50,
#     validation_data=validation_generator,
#     validation_steps=25227 // batch_size,
#     callbacks=[
#         ModelCheckpoint('data/augmented_thumb2/weights_lr-4.{epoch:02d}-{val_loss:.2f}.hdf5'),
#         CSVLogger('data/augmented_thumb2/training_lr-4.csv')
#     ]
# )

# import cv2
# import numpy as np
# import glob
# import os
# import time

# start_time = time.time()
# with open('data/augmented_thumb2/featuremaps_test/counts.csv', 'w', 1) as counts:
#     counts.write('train_id,pct_covered\n')

#     # filenames = glob.glob('Full/Train/*.jpg')
#     filenames = glob.glob('D:\\KaggleNOAASeaLions\\Test\\*.jpg')
#     for i, filename in enumerate(filenames):
#         train_id = os.path.basename(filename).split('.')[0]
#         image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
#         # Resize the image now so that 64x64 crops are really 128x128
#         image = cv2.resize(image, dsize=(0,0), fx=.5, fy=.5)
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
#                 thumbnails.append(image[y:y+64, x:x+64])
#                 coords.append((y // 32, x // 32))

#             thumbnails = np.stack(thumbnails)
#             predictions = model.predict(thumbnails)
#             for j in range(len(coords)):
#                 features[coords[j][0], coords[j][1]] = predictions[j, 0] / predictions[j].sum()

#         counts.write(train_id + ',' + str(features.sum() / features.size) + '\n')
#         features = features * 255
#         cv2.imwrite(os.path.join('data', 'augmented_thumb2', 'featuremaps_test', train_id + '.png'), features)

#     print()

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from matplotlib import pyplot as plt

validation_tuples = list(next(validation_generator) for _ in range(25227 // batch_size))
validation_images = np.vstack(validation_tuples[i][0] for i in range(len(validation_tuples)))
validation_labels = np.vstack(validation_tuples[i][1] for i in range(len(validation_tuples)))
validation_classes = np.argmax(validation_labels, axis=-1)

probs = model.predict(validation_images)
preds = np.argmax(probs, axis=-1)

mat = confusion_matrix(validation_classes, preds)
print('Confusion matrix')
print(mat)

print('AUC')
print(roc_auc_score(validation_labels[:,0], probs[:,0]))

fpr, tpr, _ = roc_curve(validation_labels[:,0], probs[:,0])

# plt.plot(fpr, tpr, color='darkorange', lw=2)
# plt.plot([0, 0], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.show()

# diff = validation_labels[:,0] - probs[:,0]
# low = np.argpartition(diff, 100)[:100]
# high = np.argpartition(diff, -100)[-100:]

# certainty = np.abs(probs[:,0] - probs[:,1])
# low_cert = np.argpartition(certainty, 100)[:100]

# f, ax = plt.subplots(10, 10, subplot_kw={'xticks': [], 'yticks': []}, dpi=120)
# f.suptitle('Predicted LION, truly EMPTY')

# for y in range(10):
#     for x in range(10):
#         ax[y,x].imshow(validation_images[low[y * 10 + x]])

# f, ax = plt.subplots(10, 10, subplot_kw={'xticks': [], 'yticks': []}, dpi=120)
# f.suptitle('Predicted EMPTY, truly LION')

# for y in range(10):
#     for x in range(10):
#         ax[y,x].imshow(validation_images[high[y * 10 + x]])

# f, ax = plt.subplots(10, 10, subplot_kw={'xticks': [], 'yticks': []}, dpi=120)
# f.suptitle('Highest uncertainty')

# for y in range(10):
#     for x in range(10):
#         ax[y,x].imshow(validation_images[low_cert[y * 10 + x]])

# plt.show()