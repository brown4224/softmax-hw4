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

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prim(z):
    return sigmoid(z) * (1 - sigmoid(z))

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

def sigmoid_prediction(X, weight_hidden, bias_hidden):
    return sigmoid(np.dot(X, weight_hidden) + bias_hidden)
def softmax_prediction(X, weight_out, bias_out):
    return softmax(np.dot(X, weight_out) + bias_out)

def nn(X, weight_hidden, bias_hidden, weight_out, bias_out ):
    hidden_layer = sigmoid_prediction(X, weight_hidden, bias_hidden)
    return softmax_prediction(hidden_layer, weight_out, bias_out)

def cost(prediction, y):
    return - sum(prediction * np.log(y) )

def error_rate(prediction, y):
    return prediction - y

def error_hidden_layer(X, weight_out, err_out):
    return  X * ( 1 - X ) * np.dot(err_out, weight_out.T)

def gradient_weight_hidden(X, err_out):
    return np.dot(X.T, err_out)

def gradient_bias_hidden(err_out):
    return sum(err_out)

def back_prop(X, weight_hidden, bias_hidden, weight_out, bias_out):
    hidden_layer = sigmoid_prediction(X, weight_hidden, bias_hidden)
    output_layer = softmax_prediction(hidden_layer, weight_out, bias_out)

    err_out = error_out(output_layer, train_label)
    g_weight_out = gradient_weight_hidden(hidden_layer, err_out)
    g_bias_out = gradient_bias_hidden(err_out)

    err_hidden = error_hidden_layer(hidden_layer, weight_out, err_out)
    g_weight_hidden = np.dot(X.T, err_hidden)
    g_bias_hidden = sum(err_hidden)
    return g_weight_hidden, g_bias_hidden, g_weight_out, g_bias_out



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


#  Hidden Layer
bias_hidden = np.random.random(1, 30)
weight_hidden = np.random.random(784, 30)

#  Out Layer
bias_out = np.random.random(1, 10)
weight_out = np.random.random(30, 10)

# OMG, this is the best function call
delta_bh = np.zeros_like(bias_hidden)
delta_wh = np.zeros_like(weight_hidden)
delta_bo = np.zeros_like(bias_out)
delta_wo = np.zeros_like(weight_out)


learn_rate = 1e-4
# momentum_term = 0.9
cycles = 500
plot = []
for i in range(cycles):
    g_weight_hidden, g_bias_hidden, g_weight_out, g_bias_out = back_prop(X, weight_hidden, bias_hidden, weight_out, bias_out)

    delta_bh = delta_bh - learn_rate * g_bias_hidden
    delta_wh = delta_wh - learn_rate * g_weight_hidden
    delta_bo = delta_bo - learn_rate * g_bias_out
    delta_wo = delta_wo - learn_rate * g_weight_out

    bias_hidden   += delta_bh
    weight_hidden += delta_wh
    bias_out      += delta_bo
    weight_out    += delta_wo

    prediction = nn(train_data, weight_hidden, bias_hidden, weight_out, bias_out )
    plot.append(cost(prediction, train_label))





