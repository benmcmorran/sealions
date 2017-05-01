"""
Multi-loss function that adds the specific task of detection
to the network. 

If creating model and transfering weights, set TRANSFER_LEARNING
to true, otherwise set the path to the saved model
"""

from keras.layers import Conv2D, Dropout, Flatten, multiply, Input, Dense
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.applications.resnet50 import ResNet50
from load_data import multi_generator

DEFINE = True                     # whether to transfer weights`
INIT_MODEL_NAME = 'model_multi_loss.h5'  # the model to transfer from
MODEL_NAME = 'model_multi_loss_soft.h5'            # the saved model


if DEFINE:
    # define functional model
    main_input = Input(shape=(416,624,3), name='main_input')
    x = conv = Conv2D(3, (3, 3), activation='relu', padding='same', input_shape=(416, 624, 3))(main_input)
    x = ResNet50(weights='imagenet', include_top=False)(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    feature_map = Dense(256, activation='relu')(x)

    # split and define detection head
    detect = Dense(64, activation='relu')(feature_map)
    detect = Dense(2, activation='softmax', name='detect_output')(detect)

    # split and define regression head
    regress = Dense(64, activation='relu')(feature_map)
    regress = Dense(5, name='regress_output')(regress)

    # merge and define final loss
    detected = Dense(1, activation='relu')(detect)
    final = multiply([detected, regress], name='final_output')
    final_model = Model(inputs=[main_input], outputs=[detect, regress, final])
else:
    final_model = load_model(MODEL_NAME)

final_model.compile(
    optimizer='adam',
    loss={'detect_output': 'binary_crossentropy', 
          'regress_output': 'mse',
          'final_output': 'mse'}, 
    loss_weights={'final_output': 1,'detect_output': 1, 'regress_output': 1},
    metrics={ 'detect_output' : 'accuracy'})

save = ModelCheckpoint(MODEL_NAME)

generator = multi_generator(10)

final_model.fit_generator(
   generator, 
   1000, # save model every 1000 samples
   epochs=25, 
   verbose=1, 
   callbacks=[save], 
   pickle_safe=True,
   class_weight={
    0:1,
    1:1}
)
