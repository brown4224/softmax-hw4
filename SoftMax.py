#  Sean McGlincy
#  HW 4
#  Reference: https://github.com/MichalDanielDobrzanski/DeepLearningPython35
import numpy as np
np.seterr(over='ignore' )

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
    # Prevent Overflow: https://stackoverflow.com/questions/23128401/overflow-error-in-neural-networks-implementation
    signal = np.clip(z, -1.0, 1.0)
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

    return sum(predict == y) / len(y)


k_fold = 2
threshold = 0.5
data = np.genfromtxt('MNIST_HW4.csv', delimiter=',', dtype=int, skip_header=1)


# Start K Folding
kf = KFold(n_splits=k_fold)
d = kf.split(data)
train_index, test_index = next(d)  #https://stackoverflow.com/questions/27380636/sklearn-kfold-acces-single-fold-instead-of-for-loop

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


cycles = 6000
plot = []
# Reference :  https://www.youtube.com/watch?v=h3l4qz76JhQ
for i in range(cycles):
    layer_0 = train_data
    layer_1 = probability(layer_0, syn0)
    layer_2 = probability(layer_1, syn1)


    err_rate = train_label - layer_2
    if(i % 1000 == 0):
        print("Error: " + str(np.mean(abs(err_rate))))
    plot.append(err_rate)

    l2_delta = err_rate * sigmoid_prim(layer_2)
    l1_delta = np.dot(l2_delta, syn1.T) * sigmoid_prim(layer_1)

    syn0 += np.dot(layer_0.T, l1_delta)
    syn1 += np.dot(layer_1.T, l2_delta)


layer_0 = test_data
layer_1 = probability(layer_0, syn0)
layer_2 = probability(layer_1, syn1)
pred = prediction(layer_2)
print(pred)
acc = accuracy(pred, test_label)
print("Accuracy: ", acc)