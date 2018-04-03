
# Sean McGlincy
# HW 4

# https://deepnotes.io/softmax-crossentropy
# https://stackoverflow.com/questions/27380636/sklearn-kfold-acces-single-fold-instead-of-for-loop
# https://github.com/MichalDanielDobrzanski/DeepLearningPython35.

import numpy as np

from sklearn.model_selection import KFold

def NormalizeData(data):
    return data / 255.0

def ArrayLabels(label):
    bit_array = np.zeros((10,1), dtype=np.int8)
    bit_array[label] = 1
    return  bit_array


k_fold = 10
data = np.genfromtxt('MNIST_HW4.csv', delimiter=',', dtype=int, skip_header=1)

# Start K Folding
kf = KFold(n_splits=k_fold)
d = kf.split(data)
train_index, test_index = next(d)

train_data = np.array(data[train_index])
test_data = np.array(data[test_index])

train_label = [ ArrayLabels(y) for y in train_data[:, 0]]
test_label =  test_data[:, 0]

train_data = NormalizeData(train_data[:, 1:])
test_data = NormalizeData(test_data[:, 1:])

train_data = [ np.reshape(data, (784, 1))  for data in train_data]
test_data = [ np.reshape(data, (784, 1))  for data in test_data]

train_data = zip(train_data, train_label)
test_data =  zip(test_data, test_label)


import network

net = network.Network([784, 30, 10])
net.SGD(train_data, 30, 10, 3.0, test_data=test_data)







