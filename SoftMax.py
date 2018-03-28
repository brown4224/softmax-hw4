#  Sean McGlincy
#  HW 4

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# np.seterr(over='ignore' )
from sklearn.model_selection import KFold

#####################################################################################################
#######################################  References   #####################################################
#####################################################################################################
# http://peterroelants.github.io/posts/neural_network_implementation_part04/
# Reference: https://github.com/MichalDanielDobrzanski/DeepLearningPython35
# https://stackoverflow.com/questions/27380636/sklearn-kfold-acces-single-fold-instead-of-for-loop
# Prevent Overflow: https://stackoverflow.com/questions/23128401/overflow-error-in-neural-networks-implementation
# Reference :  https://www.youtube.com/watch?v=h3l4qz76JhQ

#####################################################################################################
###################################  Functions   ####################################################
#####################################################################################################

#  Functions from walkthough:  http://peterroelants.github.io/posts/neural_network_implementation_part04/
def NormalizeData(data):
    return data / 255.0


def ArrayLabels(label):
    label_arr = np.zeros((len(label), 10))
    for i in range(len(label)):
        index = label[i]
        label_arr[i][index] = 1
    return label_arr

def get_z(X, w, b):

    return np.dot(w, X) + b


def sigmoid(X, w, b):
    z = get_z(X, w, b)
    # z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prim(X, w, b):
    # z = get_z(X, w, b)


    return sigmoid(X, w, b) * (1 - sigmoid(X, w, b))

def softmax(X, w, b):
    z = get_z(X, w, b)
    return np.exp(z) / np.sum(np.exp(z))



def nn(X, weight_hidden, bias_hidden, weight_out, bias_out ):
    hidden_layer = sigmoid_prediction(X, weight_hidden, bias_hidden)
    return softmax_prediction(hidden_layer, weight_out, bias_out)

def cost(prediction, y):
    return  np.multiply(y, np.log(prediction)).sum() / len(y[0])

def cost_derivative(y, layer, derviative):
    return (layer - y) * derviative


def error_hidden_layer(X, weight_out, err_out):
    return  X * ( 1 - X ) * np.dot(err_out, weight_out.T)



def back_prop(X, y,weight_hidden, bias_hidden, weight_out, bias_out):

    hidden_layer = sigmoid(X, weight_hidden, bias_hidden)
    out_layer = sigmoid(hidden_layer, weight_out, bias_out)

    s1 = sigmoid_prim(X, weight_hidden, bias_hidden)
    s2 = sigmoid_prim(hidden_layer, weight_out, bias_out)

    l2_delta = (out_layer - np.reshape(y, (10,1))) * s2
    delta_wo = np.dot(l2_delta , hidden_layer.T)

    l1_delta =  np.dot( delta_wo.T, l2_delta)  *  s1  # todo should this be the input layer?
    # delta_bh =  np.dot(out_layer.T, delta_bo)  * sigmoid_prim(X, weight_hidden, bias_hidden)  # todo should this be the input layer?
    # delta_bh =  np.dot(delta_bo, np.reshape(X, (1, 784)) ).T * sigmoid_prim(X, weight_hidden, bias_hidden)  # todo should this be the input layer?

    print(np.dot(delta_wo.T, l2_delta).shape)
    print(sigmoid_prim(X, weight_hidden, bias_hidden).shape)



    # delta_wh =  np.dot(delta_bh, np.reshape(X, (1, 784)).T)   # todo should this be the input layer?
    # delta_wh = np.dot(delta_bh, X)
    # delta_wh = np.dot(delta_bh, X.T)

    print("s1", s1.shape)
    print(s2.shape)
    print(l2_delta.shape)
    print(l1_delta.shape)
    print(X.T.shape)
    exit(1)
    delta_wh = np.dot(l1_delta , X.T)

    return delta_wh, l1_delta, delta_wo, l2_delta








def DisplayLearningCurve(plot):
    plt.plot(plot)
    plt.interactive(False)
    plt.show(block=True)


def prediction(prob):
    return np.argmax(prob, axis=1)
def accuracy(predict, y):
    p = prediction(predict)
    return sum(p == y) / len(y)
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
test_label= ArrayLabels(test_data[:, 0])
# test_label = test_data[:, 0]

train_data = NormalizeData(train_data[:, 1:])
test_data = NormalizeData(test_data[:, 1:])


#  [784, 30, 10]
np.random.seed(1)
small_val = 0.1


#  Hidden Layer
weight_hidden = np.random.random((30, 784)) * small_val
bias_hidden = np.random.random((30, 1 )) * small_val


#  Out Layer
# weight_out = np.random.random((30, 10)) * small_val
# bias_out = np.random.random((1, 10)) * small_val
weight_out = np.random.random((10, 30)) * small_val
bias_out = np.random.random((10, 1)) * small_val

# #  Hidden Layer
# weight_hidden = np.random.random((784, 30)) * small_val
# bias_hidden = np.random.random((784, 1)) * small_val
#
#
# #  Out Layer
# weight_out = np.random.random((30, 10)) * small_val
# bias_out = np.random.random((30, 1)) * small_val


# OMG, this is the best function call
delta_wh = np.zeros_like(weight_hidden)
delta_bh = np.zeros_like(bias_hidden)
delta_wo = np.zeros_like(weight_out)
delta_bo = np.zeros_like(bias_out)


lr = 1e-4
cycles = 1
plot = []



for i in range(cycles):
    for X, Y in zip(train_data, train_label):
        g_weight_hidden, g_bias_hidden, g_weight_out, g_bias_out = back_prop(X, Y , weight_hidden,
                                                                             bias_hidden, weight_out, bias_out)
        weight_hidden =  weight_hidden - lr * g_weight_hidden
        bias_hidden =  bias_hidden - lr * g_bias_hidden

        weight_out =  weight_out - lr * g_weight_out
        bias_out =  bias_out - lr * g_bias_out



# DisplayLearningCurve(plot)
#
# pred = nn(test_data, weight_hidden, bias_hidden, weight_out, bias_out)
# acc = accuracy(pred, test_label)
# print(acc)