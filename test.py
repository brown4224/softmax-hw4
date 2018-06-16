
# Sean McGlincy
# HW 4

# https://github.com/uiureo/nn/blob/master/network.py
# https://stats.stackexchange.com/questions/153285/derivative-of-softmax-and-squared-error
# https://susanqq.github.io/tmp_post/2017-09-05-crossentropyvsmes/
# https://deepnotes.io/softmax-crossentropy
# https://stackoverflow.com/questions/33541930/how-to-implement-the-softmax-derivative-independently-from-any-loss-function
# http://peterroelants.github.io/posts/neural_network_implementation_part05/
# http://peterroelants.github.io/posts/neural_network_implementation_part04/
# https://deepnotes.io/softmax-crossentropy
# https://stackoverflow.com/questions/27380636/sklearn-kfold-acces-single-fold-instead-of-for-loop
# https://github.com/MichalDanielDobrzanski/DeepLearningPython35.

import numpy as np
import network
from sklearn.model_selection import KFold

# Normalizes Data between 0-> 1
def NormalizeData(data):
    return data / 255.0

# One Hot array for training labels
# EX.   [0,0,0,0,1,0,0,0,0,0]
def ArrayLabels(label):
    bit_array = np.zeros((10,1), dtype=np.int8)
    bit_array[label] = 1
    return  bit_array


k_fold = 10
data = np.genfromtxt('MNIST_HW4.csv', delimiter=',', dtype=int, skip_header=1)

# K Fold for one cycle
kf = KFold(n_splits=k_fold)
d = kf.split(data)
train_index, test_index = next(d)

# Kfold map
train_data = np.array(data[train_index])
test_data = np.array(data[test_index])

# Prep training and test labels
# Training labels are one hot array
train_label = [ ArrayLabels(y) for y in train_data[:, 0]]
test_label =  test_data[:, 0]

# Prep training and test data
train_data = NormalizeData(train_data[:, 1:])
test_data = NormalizeData(test_data[:, 1:])

# Reshape data to an array of [784, 1]
train_data = [ np.reshape(data, (784, 1))  for data in train_data]
test_data = [ np.reshape(data, (784, 1))  for data in test_data]

# Zip (x,y)
train_data = zip(train_data, train_label)
test_data =  zip(test_data, test_label)

net = network.Network([784, 45, 10])
net.SGD(train_data, 60, 10, 0.5, test_data=test_data)







