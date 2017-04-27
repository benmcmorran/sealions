from keras.models import Sequential
from keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from load_data import train_generator, train_sliced_generator, train_sliced
from keras.applications.resnet50 import ResNet50

"""
    A straightforward model that hopes and prays conv nets are powerful enough
    to connect the counts of different seal types with the little seals.

    (untested - but compiles and runs)
    note: currently assumes small set of data.
"""
model = Sequential([
    Conv2D(32, kernel_size=(2, 2),
                 activation='relu',
                 input_shape=(416, 624, 3)),
    Conv2D(64, (2, 2), activation='relu'),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5)
]);

model.summary()

model.compile(optimizer='adam',
              loss='mse', 
              metrics=['accuracy'])

#generator = train_sliced_generator(2)
import numpy as np
x, y = train_sliced(4)
x_tr, y_tr = x[2:,:,:,:], y[2:,:]
x_val, y_val = x[2:,:,:,:], y[2:,:]

model.fit(x_tr, y_tr, batch_size=1, epochs=5, validation_data=(x_val, y_val))

# model.fit_generator(
#    generator, 
#    2, # sample per epoch 
#    epochs=1, 
#    verbose=1, 
#    callbacks=None, 
#    validation_data=None,
# )

model.save('model_sliced_resnet_regressor_model.h5')
