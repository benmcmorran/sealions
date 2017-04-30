from keras.models import Sequential
from keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.models import load_model
from load_data import train_generator, train_sliced_generator, train_sliced
from keras.applications.resnet50 import ResNet50
from keras import regularizers

"""
    A straightforward model that hopes and prays conv nets are powerful enough
    to connect the counts of different seal types with the little seals.

    (untested - but compiles and runs)
    note: currently assumes small set of data.
"""
model = load_model('model_target_res_reg.h5')
"""

model = Sequential()
model.add(Conv2D(3, (3, 3), activation='relu', padding='same', input_shape=(416, 624, 3)))
model.add(ResNet50(weights='imagenet', include_top=False))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5, kernel_regularizer=regularizers.l2(0.001)))
model.load_weights('model_target_res_1.h5')
model.summary()

model.compile(optimizer='adam',
              loss='mse', 
              metrics=['accuracy'])
"""
generator = train_sliced_generator(6)
import numpy as np
x, y = train_sliced(99)
print np.shape(x)
print np.shape(y)
x_val, y_val   = x[0:5,:,:,:], y[0:5,:]

img_names = [ '490.jpg' ]
# x_tr, y_tr = x[5:,:,:,:], y[5:,:]
# model.fit(x_tr, y_tr, batch_size=5, epochs=150, validation_data=(x_val, y_val))
# 65082
model.fit_generator(
    generator, 
    10967, # batches per epoch 
    epochs=10,
    verbose=1,
    callbacks=None, 
    validation_data=(x_val, y_val)
)

model.save('model_target_full_res.h5')
