# A python script that takes an image file containing a handwritten digit and identifies 
# the digit using a supervised learning algorithm and the MNIST dataset.

# Import tensorflow as tf
import tensorflow as tf
# Getting rid of warnings
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

# Import input_data
from tensorflow.examples.tutorials.mnist import input_data

# Store the image data in the variable mnist 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # y labels are oh-encoded

# Size of the dataset that was just imported. 
# Looking at the num_examples for each of the three subsets, we can determine that the dataset 
# has been split into 55,000 images for training, 5000 for validation, and 10,000 for testing.
n_train = mnist.train.num_examples # 55,000
n_validation = mnist.validation.num_examples # 5000
n_test = mnist.test.num_examples # 10,000

tf.logging.set_verbosity(old_v)