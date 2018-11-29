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

# Imports for image manipulation
import numpy as np
from PIL import Image

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


# Training and Testing

# Define method of evaluating the accuracy so it can be printed out on mini-batches of data while it trains. These printed 
# statements will allow to check that from the first iteration to the last, loss decreases and accuracy increases; 
# they will also allow to track whether or not it has ran enough iterations to reach a consistent and optimal result
correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#  Initialize a session for running the graph. This session will feed the network with training examples, and once trained, 
# same graph can be feeded with new test examples to determine the accuracy of the model
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# At each training step, the parameters are adjusted slightly to try and reduce the loss for the next step. As the 
# learning progresses, reduction in loss should should be ssen, and eventually it can stop training and use the 
# network as a model for testing new data

# Train on mini batches
for i in range(n_iterations):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={X: batch_x, Y: batch_y, keep_prob:dropout})

    # Print loss and accuracy (per minibatch)
    if i%100==0:
        minibatch_loss, minibatch_accuracy = sess.run([cross_entropy, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob:1.0})
        print("Iteration", str(i), "\t| Loss =", str(minibatch_loss), "\t| Accuracy =", str(minibatch_accuracy))

# Once the training is complete, can run the session on the test images. This time keep_prob dropout rate of 1.0 is used to
# ensure all units are active in the testing process
test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob:1.0})
print("\nAccuracy on test set:", test_accuracy)

while True:
    # Get image file
    imageName = input('Name of the file(eg: 1 or eg: 2): ')
    imageName = "Images/data/" + imageName + ".png"
    print(imageName)
    img = np.invert(Image.open(imageName).convert('L')).ravel()

    # Now that the image data is structured correctly, can run a session in the same way as previously, but this time
    # only feeding in the single image for testing
    prediction = sess.run(tf.argmax(output_layer,1), feed_dict={X: [img]})
    print ("Prediction for test image:", np.squeeze(prediction))

    # Continue or exit
    sentinel = input('Would you like to continue? (type "yes" or "no"): ')

    if sentinel == "no":
        break