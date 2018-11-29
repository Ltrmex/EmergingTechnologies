# Student: Maciej Majchrzak
# Student Number: G00332746
# References: https://www.digitalocean.com/community/tutorials/how-to-build-a-neural-network-to-recognize-handwritten-digits-with-tensorflow

# A python script that takes an image file containing a handwritten digit and identifies 
# the digit using a supervised learning algorithm and the MNIST dataset.

# Import tensorflow as tf
import tensorflow as tf
# Getting rid of warnings
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

# Importing the MNIST Dataset
from tensorflow.examples.tutorials.mnist import input_data

# Store the image data in the variable mnist 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # y labels are oh-encoded

# Size of the dataset that was just imported. 
# Looking at the num_examples for each of the three subsets, we can determine that the dataset 
# has been split into 55,000 images for training, 5000 for validation, and 10,000 for testing.
n_train = mnist.train.num_examples # 55,000
n_validation = mnist.validation.num_examples # 5000
n_test = mnist.test.num_examples # 10,000

# Defining the Neural Network Architecture

# Store the number of units per layer in global variables
n_input = 784   # input layer (28x28 pixels)
n_hidden1 = 512 # 1st hidden layer
n_hidden2 = 256 # 2nd hidden layer
n_hidden3 = 128 # 3rd hidden layer
n_output = 10   # output layer (0-9 digits)

# Hyperparameters which will stay constant
learning_rate = 1e-4
n_iterations = 1000
batch_size = 128
dropout = 0.5

# Building the TensorFlow Graph

# Defining three tensors as placeholders, which are tensors that we'll feed values into later
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])
keep_prob = tf.placeholder(tf.float32) 

# The weights and bias tensors are stored in dictionary objects for ease of access.

# Random values from a truncated normal distribution for the weights. We want them to be close to zero, so they 
# can adjust in either a positive or negative direction, and slightly different, so they generate different errors. 
# This will ensure that the model learns something useful.
weights = {
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
    'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
}

# For the bias, use a small constant value to ensure that the tensors activate in the intial stages and therefore 
# contribute to the propagation
biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
}

# Set up the layers of the network by defining the operations that will manipulate the tensors
# Each hidden layer will execute matrix multiplication on the previous layer’s outputs and the current layer’s weights, 
# and add the bias to these values
layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
layer_drop = tf.nn.dropout(layer_3, keep_prob)
output_layer = tf.matmul(layer_3, weights['out']) + biases['out']

# There are several choices of gradient descent optimization algorithms already implemented in TensorFlow, 
# but following code uses the Adam optimizer.  This extends upon gradient descent optimization by using momentum to speed 
# up the process through computing an exponentially weighted average of the gradients and using that in the adjustments
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

