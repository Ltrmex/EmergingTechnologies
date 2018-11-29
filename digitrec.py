# A python script that takes an image file containing a handwritten digit and identifies 
# the digit using a supervised learning algorithm and the MNIST dataset.

# Imports
import numpy as np
import math

# Building blocks of neural networks 
class Neuron(object):
   def __init__(self):
       self.weights = np.array([1.0, 2.0])
       self.bias = 0.0

   def forward(self, inputs):
       # Assuming that inputs and weights are 1-D numpy arrays and the bias is a number
       a_cell_sum = np.sum(inputs * self.weights) + self.bias

       # This is the sigmoid activation function
       result = 1.0 / (1.0 + math.exp(-a_cell_sum)) 
       return result

neuron = Neuron()
output = neuron.forward(np.array([1,1]))

print(output)

# Retrieving training and test data

# Import Numpy, keras and MNIST data
import numpy as np 
import matplotlib.pyplot as plt 
from keras.datasets import mnist
from keras.models import Sequential 
from keras.layers.core import Dense,Dropout,Activation
from keras.utils import np_utils

# Retrieving the training and test data
(X_train,y_train),(X_test,y_test)=mnist.load_data()
print('X_train shape:',X_train.shape)
print('X_test shape: ',X_test.shape)
print('y_train shape:',y_train.shape)
print('y_test shape: ',y_test.shape)