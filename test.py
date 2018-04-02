
import numpy as np

from sklearn.model_selection import KFold

def NormalizeData(data):
    return data / 255.0

def ArrayLabels(label):
    bit_array = np.zeros((10,1), dtype=np.int8)
    bit_array[label] = 1
    return  bit_array
    # label_arr = np.zeros((len(label), 10))
    # for i in range(len(label)):
    #     index = label[i]
    #     label_arr[i][index] = 1.0
    # return label_arr


k_fold = 10
data = np.genfromtxt('MNIST_HW4.csv', delimiter=',', dtype=int, skip_header=1)

# Start K Folding
kf = KFold(n_splits=k_fold)
d = kf.split(data)
#https://stackoverflow.com/questions/27380636/sklearn-kfold-acces-single-fold-instead-of-for-loop
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



# train_data = [ [x.reshape(784, 1), y]  for x,y in zip(train_data, train_label)]
# test_data = [ [x.reshape(784, 1), y]  for x,y in zip(test_data, test_label)]


# - network.py example:
import network

net = network.Network([784, 30, 10])
net.SGD(train_data, 30, 10, 3.0, test_data=test_data)







