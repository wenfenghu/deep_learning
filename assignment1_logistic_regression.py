import numpy as np
import os
import sys

from sklearn.linear_model import LogisticRegression
from six.moves import cPickle as pickle
from sklearn import metrics
from time import time


def reshape(array):
    nsamples, nx, ny = array.shape
    d2_train_dataset = array.reshape((nsamples, nx * ny))
    return d2_train_dataset

# Load the dictionary back from the pickle file.

raw_data = pickle.load(open("notMNIST.pickle", "rb"))

train_dataset = reshape(raw_data['train_dataset'])
train_labels = raw_data['train_labels']
valid_dataset = reshape(raw_data['valid_dataset'])
valid_labels = raw_data['valid_labels']
#'test_dataset': test_dataset,
#'test_labels': test_labels,

# get the model
logistic = LogisticRegression()

# Fit the model using training dataset
t0 = time()
logistic.fit(train_dataset, train_labels)
print("Training model time of", len(train_labels), "is", round(time()-t0, 3) / 60, "m")

# predict the validation dataset
predicted = logistic.predict(valid_dataset)

# print out the prediction accuracy
print( "prediction accuracy = ", metrics.accuracy_score(valid_labels, predicted))