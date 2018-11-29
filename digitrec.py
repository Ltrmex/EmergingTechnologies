# A python script that takes an image file containing a handwritten digit and identifies 
# the digit using a supervised learning algorithm and the MNIST dataset.

import numpy as np
import math

class Neuron(object):
   def __init__(self):
       self.weights = np.array([1.0, 2.0])
       self.bias = 0.0
   def forward(self, inputs):
       """ Assuming that inputs and weights are 1-D numpy arrays and the bias is a number """
       a_cell_sum = np.sum(inputs * self.weights) + self.bias
       result = 1.0 / (1.0 + math.exp(-a_cell_sum)) # This is the sigmoid activation function
       return result
neuron = Neuron()
output = neuron.forward(np.array([1,1]))
print(output)