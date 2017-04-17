from keras.models import Sequential
from keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from load_data import train_generator
from keras.applications.resnet50 import ResNet50

"""
    A straightforward model that hopes and prays conv nets are powerful enough
    to connect the counts of different seal types with the little seals.

    (untested - but compiles and runs)
    note: currently assumes small set of data.
"""
model = Sequential()
model.add(Conv2D(3, (3, 3), activation='relu', padding='same', input_shape=(4992, 3328, 3)))
model.add(ResNet50(weights='imagenet', include_top=False))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))

model.summary()

model.compile(optimizer='adam',
              loss='mse')

generator = train_generator(5)

model.fit_generator(
    generator, 
    10, # sample per epoch 
    epochs=1, 
    verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_q_size=10, workers=1, pickle_safe=False, initial_epoch=0
)

model.save('model_resnet_regressor_model.h5')


