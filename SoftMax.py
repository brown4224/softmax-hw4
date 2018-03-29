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

    # print(X.shape)
    # print(w.shape)
    return np.dot(w, X) + b
    # return np.dot( X, w) + b


def sigmoid(z):

    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prim(z):
    return sigmoid(z) * (1 - sigmoid(z))





def nn(X, wh, bh, wo, bo):
    z = get_z(X, wh, bh)
    hidden_layer = sigmoid(z)
    z = get_z(hidden_layer, wo, bo)
    return sigmoid(z)

def cost(p, y):
    return np.sum((p - y)**2) / len(y[0])
    # return (np.sum(p - y)**2)**0.5 / len(y)
    # return np.mean(np.abs(p - y))

# def acuraacy(p, y):
#     y = prediction(y)
#     p = prediction(p)
#     return mean(abs(p == y))

# def error_hidden_layer(X, weight_out, err_out):
#     return  X * ( 1 - X ) * np.dot(err_out, weight_out.T)



def back_prop(X, y, wh, bh, wo, bo):

    layer_0 = X
    z = get_z(layer_0, wh, bh )
    layer_1 = sigmoid(z)
    s1 = sigmoid_prim(z)
    z = get_z(layer_1, wo, bo )
    layer_2 = sigmoid(z)
    s2 = sigmoid_prim(z)

    l2_delta = (layer_2 - y) * s2
    l1_delta = np.dot( wo.T, l2_delta) * s1
    # l1_delta = np.dot(  l2_delta, wo.T) * s1

    wh_delta = np.dot( l1_delta, layer_0.T)
    wo_delta = np.dot(l2_delta, layer_1.T)

    # print(l1_delta.shape)
    # print(l2_delta.shape)
    # print(wh.shape)
    # print(wo.shape)
    # print("wh delta: ", wh_delta[0])
    # print("wo delta: ", wo_delta[0])

    return wh_delta, l1_delta, wo_delta, l2_delta








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
# test_label= ArrayLabels(test_data[:, 0])
test_label = test_data[:, 0]

train_data = NormalizeData(train_data[:, 1:])
test_data = NormalizeData(test_data[:, 1:])


#  [784, 30, 10]
np.random.seed(1)
small_val = 1

#  Hidden Layer
weight_hidden = np.random.random((30, 784)) * small_val
bias_hidden = np.random.random((30, 1 )) * small_val


#  Out Layer
weight_out = np.random.random((10, 30,)) * small_val
bias_out = np.random.random((10, 1)) * small_val
# #  Hidden Layer
# weight_hidden = np.random.random((784, 30)) * small_val
# bias_hidden = np.random.random((1, 30 )) * small_val
#
#
# #  Out Layer
# weight_out = np.random.random((30, 10,)) * small_val
# bias_out = np.random.random((1, 10)) * small_val



# OMG, this is the best function call
delta_wh = np.zeros_like(weight_hidden)
delta_bh = np.zeros_like(bias_hidden)
delta_wo = np.zeros_like(weight_out)
delta_bo = np.zeros_like(bias_out)


lr = 1e-2
cycles = 1000
plot = []


j = 0
for i in range(cycles):
    for X, Y in zip(train_data, train_label):
        j += 1
        X = np.reshape(X, (784,1))
        Y = np.reshape(Y, (10,1))
        g_weight_hidden, g_bias_hidden, g_weight_out, g_bias_out = back_prop(X, Y , weight_hidden,
                                                                         bias_hidden, weight_out, bias_out)

        # g_weight_hidden, g_bias_hidden, g_weight_out, g_bias_out = back_prop(train_data, train_label, weight_hidden,
    #                                                                      bias_hidden, weight_out, bias_out)

    # print("Bias", g_bias_hidden)
    # print("Weight", g_weight_hidden)

        weight_hidden =  weight_hidden - lr * g_weight_hidden
        bias_hidden =  bias_hidden - lr * g_bias_hidden

        weight_out =  weight_out - lr * g_weight_out
        bias_out =  bias_out - lr * g_bias_out

        if j % 100 == 0:
            predict = nn(X, weight_hidden, bias_hidden, weight_out, bias_out)
            c = cost(predict, Y)
            plot.append(c)
            # print(c)
DisplayLearningCurve(plot)


# print(test_data.shape)
# print(weight_hidden.shape)
# print(bias_hidden.shape)
#
# predict = nn(test_data, weight_hidden, bias_hidden, weight_out, bias_out)
# acc = accuracy(predict,test_label )
# print("Accuracy: ", acc)

#
# pred = nn(test_data, weight_hidden, bias_hidden, weight_out, bias_out)
# acc = accuracy(pred, test_label)
# print(acc)