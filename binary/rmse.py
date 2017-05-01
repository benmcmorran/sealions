import numpy as np

def rmse(truth, prediciton):
    return np.sqrt(np.square(truth - prediciton).mean(axis=1)).mean()

train = np.loadtxt('Full/Train/train.csv', delimiter=',', skiprows=1)[:,1:]
pred = np.loadtxt('pred2.csv', delimiter=',', skiprows=1)[:,1:]

print('Zero baseline: {}'.format(rmse(train, np.zeros(train.shape))))
print('Mean baseline: {}'.format(rmse(train, train.mean(axis=0))))
print('Prediciton: {}'.format(rmse(train, pred)))