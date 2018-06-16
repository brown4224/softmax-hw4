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
# %load network.py


"""
network.py
~~~~~~~~~~
IT WORKS

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.plot = []

    # Feed forward using sigmoid as fist activation layer and softmax as output layer
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        a = sigmoid(np.dot(self.weights[0], a) + self.biases[0])
        return softmax(np.dot(self.weights[1], a) + self.biases[1])


    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        # Runs mini batches for the number of epochs
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test));
            else:
                print("Epoch {} complete".format(j))

            # Calculate the Cost after each mini batch using MSE
            c = self.get_cost_mse(test_data)
            self.plot.append(c)

        # Display the cost from the plot array
        # This allows us to see the gradient curve
        DisplayLearningCurve(self.plot)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer

        #  Calculates feed forward for each layer
        #  This loop does two iteration using the sigmoid function.
        #  Replace with softmax after loop terminates.
        for b, w in zip(self.biases, self.weights):

            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        ##########################################################
        ##############    Changes  ###############################
        ##########################################################

        # Update the author's code: Feed Forward NN
        # Update the last function to be softmax instead of sigmoid
        m = y.shape[1]
        s = softmax(zs[-1])
        x = s
        activations[-1] = s

        # Chain Rule
        # Find Delta between Hidden layer and output layer
        # Use Soft Max Partial derivative  (element wise multiplication) MSE Partial derivative.
        delta = (x - y) * softmax_prime(s)


        ##########################################################
        ##########################################################
        ##########################################################

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.

        for l in range(2, self.num_layers):

            z = zs[-l]
            sp =  sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


    # This function is the derivative of softmax and  cross entropy
    def cost_derivative(self, x, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return  x - y

    #  This is the cross entropy cost
    #  Usally has a larger decision boundry they MSE
    def cost_cross_entropy(self, y_prediction, y):
        likilihood = -1 * y * np.log(y_prediction)
        return np.sum(likilihood)/ y.shape[0]

    # Mean Square error
    def mean_err_sq(self, x, y):
        return 0.5 * np.mean(x - y) ** 2

    def get_cost_mse(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(self.mean_err_sq(x,y) for (x, y) in test_results)



def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


#  This is the stable Softmax function
#  It bit shifts to avoid "nan"
def softmax(z):
    exp = np.exp(z - np.max(z))
    return exp / np.sum(exp)

# Softmax Prime
# Assumes that Softmax has been passed in as value
def softmax_prime(s):
    return s * (1- s)


#  Displays the Graph
def DisplayLearningCurve(plot):
    plt.plot(plot)
    plt.interactive(False)
    plt.show(block=True)