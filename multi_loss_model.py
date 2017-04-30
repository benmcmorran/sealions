"""
Multi-loss function that adds the specific task of detection
to the network. 

If creating model and transfering weights, set TRANSFER_LEARNING
to true, otherwise set the path to the saved model
"""
from keras.layers import Dropout, Flatten, multiply, Input, Dense
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model

from load_data import multi_generator

TRANSFER_LEARNING = False          # whether to transfer weights
INIT_MODEL_NAME = 'trash.h5'       # the model to transfer from
MODEL_NAME = 'model_multi_loss.h5' # the saved model

if TRANSFER_LEARNING:
    # load old model
    old_model = load_model(INIT_MODEL_NAME)

    # load specific weights
    res_weights = old_model.layers[0].get_weights()
    last_weights = old_model.layers[-1].get_weights()

    # define functional model
    main_input = Input(shape=(416,624,3), name='main_input')
    x = old_model.layers[0]
    x.set_weights(res_weights)
    x = x(main_input)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    feature_map = Dense(64, activation='relu')(x)

    # split and define detection head
    detect = Dense(64, activation='relu')(feature_map)
    detect = Dense(1, activation='softmax', name='detect_output')(detect)

    # split and define regression head
    regress = Dense(64, activation='relu')(feature_map)
    regress = Dense(5, name='regress_output')(regress)

    # merge and define final loss
    final = multiply([detect, regress], name='final_output')
    final_model = Model(inputs=[main_input], outputs=[detect, regress, final])
else:
    final_model = load_model(MODEL_NAME)

final_model.compile(
    optimizer='adam',
    loss={'detect_output': 'binary_crossentropy', 
          'regress_output': 'mse',
          'final_output': 'mse'}, #
    loss_weights={'final_output': 2,'detect_output': 1, 'regress_output': 1})

save = ModelCheckpoint('model_multi_loss.h5')

generator = multi_generator(6)

final_model.fit_generator(
   generator, 
   1, # sample per epoch 
   epochs=1, 
   verbose=1, 
   callbacks=[save], 
   pickle_safe=True,
   class_weight={
    0:10,
    1:1}
)