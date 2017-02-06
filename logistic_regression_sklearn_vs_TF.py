import numpy as np
import os
import sys

from sklearn.linear_model import LogisticRegression
from six.moves import cPickle as pickle
from sklearn import metrics
from time import time
import tensorflow as tf

### can be removed! ###
def reshape(array):
    nsamples, nx, ny = array.shape
    d2_train_dataset = array.reshape((nsamples, nx * ny))
    return d2_train_dataset

def reformat(dataset, labels):
    image_size = 28
    num_labels = 10
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

def logisticRegression(train_dataset, train_labels, dataset):
    # get the LogisticRegression model
    logistic = LogisticRegression()

    # Fit the model using training dataset
    t0 = time()
    logistic.fit(train_dataset, train_labels)
    print("Training model time of", len(train_labels), "is", round(time()-t0, 3) / 60, "m")

    # predict the validation dataset
    predicted_labels = logistic.predict(dataset)
    return predicted_labels


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


# Load the dictionary back from the pickle file.
raw_data = pickle.load(open("notMNIST.pickle", "rb"))

train_dataset = raw_data['train_dataset']
train_labels = raw_data['train_labels']
valid_dataset = raw_data['valid_dataset']
valid_labels = raw_data['valid_labels']
test_dataset = raw_data['test_dataset']
test_labels = raw_data['test_labels']

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

del raw_data # to free memories

image_size = 28
num_labels = 10
train_subset = 100000
num_steps = 1001

###### comparision between logisticRegression using sklearn AND SGD using Tensorflow neoraul networks
### PART 1: logisticRegression using sklearn ###

# predicted = logisticRegression(train_dataset, train_labels, valid_dataset)
## print out the prediction accuracy
# print( "prediction accuracy = ", metrics.accuracy_score(valid_labels, predicted))

# ### PART 2.1:  Tensorflow GD  ###

#
# graph = tf.Graph()
# with graph.as_default():
#     # Input data.
#     # Load the training, validation and test data into constants that are
#     # attached to the graph.
#     tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
#     tf_train_labels = tf.constant(train_labels[:train_subset])
#     tf_valid_dataset = tf.constant(valid_dataset)
#     tf_test_dataset = tf.constant(test_dataset)
#
#     # Variables.
#     # These are the parameters that we are going to be training. The weight
#     # matrix will be initialized using random values following a (truncated)
#     # normal distribution. The biases get initialized to zero.
#     weights = tf.Variable(
#         tf.truncated_normal([image_size * image_size, num_labels]))
#     biases = tf.Variable(tf.zeros([num_labels]))
#
#     # Training computation.
#     # We multiply the inputs with the weight matrix, and add biases. We compute
#     # the softmax and cross-entropy (it's one operation in TensorFlow, because
#     # it's very common, and it can be optimized). We take the average of this
#     # cross-entropy across all training examples: that's our loss.
#     logits = tf.matmul(tf_train_dataset, weights) + biases
#     loss = tf.reduce_mean(
#         tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
#
#     # Optimizer.
#     # We are going to find the minimum of this loss using gradient descent.
#     optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#
#     # Predictions for the training, validation, and test data.
#     # These are not part of training, but merely here so that we can report
#     # accuracy figures as we train.
#     train_prediction = tf.nn.softmax(logits)
#     valid_prediction = tf.nn.softmax(
#         tf.matmul(tf_valid_dataset, weights) + biases)
#     test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
#
#
# with tf.Session(graph=graph) as session:
#   # This is a one-time operation which ensures the parameters get initialized as
#   # we described in the graph: random weights for the matrix, zeros for the
#   # biases.
#   tf.global_variables_initializer().run()
#   print('Initialized')
#
#   t1 = time()
#   for step in range(num_steps):
#     # Run the computations. We tell .run() that we want to run the optimizer,
#     # and get the loss value and the training predictions returned as numpy
#     # arrays.
#     _, l, predictions = session.run([optimizer, loss, train_prediction])
#     if (step % 200 == 0):
#       print('Loss at step %d: %f' % (step, l))
#       print('Training accuracy: %.1f%%' % accuracy(
#         predictions, train_labels[:train_subset, :]))
#       # Calling .eval() on valid_prediction is basically like calling run(), but
#       # just to get that one numpy array. Note that it recomputes all its graph
#       # dependencies.
#       print('Validation accuracy: %.1f%%' % accuracy(
#         valid_prediction.eval(), valid_labels))
#   print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
#
#   print("Training model + prediction time of", train_subset, "is", round(time() - t1, 3) / 60, "m")


### results of 10000 trainning sets, 1000 epics
# Training accuracy: 77.8%
# Validation accuracy: 77.7%
# Test accuracy: 84.0%
# Training model + prediction time of 100000 is 12.553 m

### PART 2.1:  Tensorflow SGD  ###
batch_size = 128
num_steps = 3001

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weights = tf.Variable(
        tf.truncated_normal([image_size * image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
        tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)


with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")

  t2 = time()
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
  print("Training model + prediction time of", train_subset, "is", round(time() - t2, 3) / 60, "m")

# Minibatch accuracy: 82.0%
# Validation accuracy: 78.1%
# Test accuracy: 85.3%
# Training model + prediction time of 100000 is 0.13676666666666665 m