#  Sean McGlincy
#  HW 4

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# np.seterr(over='ignore' )

#####################################################################################################
#######################################  References   #####################################################
#####################################################################################################
#  Reference: https://github.com/MichalDanielDobrzanski/DeepLearningPython35
#https://stackoverflow.com/questions/27380636/sklearn-kfold-acces-single-fold-instead-of-for-loop
# Prevent Overflow: https://stackoverflow.com/questions/23128401/overflow-error-in-neural-networks-implementation
# Reference :  https://www.youtube.com/watch?v=h3l4qz76JhQ

#####################################################################################################
###################################  Functions   ####################################################
#####################################################################################################

from sklearn.model_selection import KFold

def NormalizeData(data):
    return data / 255.0


def ArrayLabels(label):
    label_arr = np.zeros((len(label), 10))
    for i in range(len(label)):
        index = label[i]
        label_arr[i][index] = 1
    return label_arr

def sigmoid(z):

    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prim(z):
    return sigmoid(z) * (1 - sigmoid(z))

def probability(layer, synapses):
    return sigmoid(np.dot(layer, synapses))
def feed_forward(X, weights, bias):
    return sigmoid(np.dot(X, weights) + b)
def prediction(prob):
    return np.argmax(prob, axis=1)
def accuracy(predict, y):
    p = prediction(predict)
    return sum(p == y) / len(y)


def Cost(prediction, y):
    label_1 = -y* np.log(prediction)            # If y=1 use this equation
    label_0 = -(1 - y)* np.log(1 - prediction)  # If y=0 use this equation
    return sum(label_1 + label_0 ) / len(y)

def DisplayLearningCurve(plot):
    plt.plot(plot)
    plt.interactive(False)
    plt.show(block=True)

#####################################################################################################
#######################################  Main   #####################################################
#####################################################################################################
k_fold = 10
threshold = 0.5
data = np.genfromtxt('MNIST_HW4.csv', delimiter=',', dtype=int, skip_header=1)


# Start K Folding
kf = KFold(n_splits=k_fold)
d = kf.split(data)
train_index, test_index = next(d)

#  Prep Data
train_data = np.array(data[train_index])
test_data = np.array(data[test_index])
train_label= ArrayLabels(train_data[:, 0])
test_label = test_data[:, 0]

train_data = NormalizeData(train_data[:, 1:])
test_data = NormalizeData(test_data[:, 1:])


#  [784, 30, 10]
np.random.seed(1)
syn0 = 2*np.random.random((784, 30)) -1
syn1 = 2*np.random.random((30, 10)) -1


cycles = 3000
plot = []
for i in range(cycles):
    layer_0 = train_data
    layer_1 = probability(layer_0, syn0)
    layer_2 = probability(layer_1, syn1)


    err_rate = train_label - layer_2
    if(i % 1000 == 0):
        print("Error: " + str(np.mean(abs(err_rate))))
    plot.append(err_rate)

    # plot.append(Cost(layer_2, train_label))


    l2_delta = err_rate * sigmoid_prim(layer_2)
    l1_delta = np.dot(l2_delta, syn1.T) * sigmoid_prim(layer_1)

    syn0 += np.dot(layer_0.T, l1_delta)
    syn1 += np.dot(layer_1.T, l2_delta)

#DisplayLearningCurve(plot)

layer_0 = test_data
layer_1 = probability(layer_0, syn0)
layer_2 = probability(layer_1, syn1)

acc = accuracy(layer_2, test_label)
print("Accuracy: ", acc)