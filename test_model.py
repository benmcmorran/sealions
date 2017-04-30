from load_data import train_generator, train_sliced_generator, train_sliced
from keras.models import load_model

MODEL_PATH = 'model_target_full_res.h5'

model = load_model(MODEL_PATH)

model.summary()

x, y = train_sliced(100)

res  = model.evaluate(x, y)
print res
pred = model.predict(x)

for i in xrange(100):
    print y[i], [int(pred[i][j]) for j in xrange(len(pred[i]))]


