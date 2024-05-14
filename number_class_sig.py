# This has been created by William Peters on the 14/05/2024
# Use at your own risk and ensure you do your own due dilligence

# Import neccessary modules for the project

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Convert read in data from csv using pandas module
data = pd.read_csv('mnist_train.csv')
#print(data.head())

# Convert data to a numpy array - this will make it easy to do matrix multiplication when it comes to forwards and bacwards propogation
data = np.array(data)

# Determine number of rows and also columns
#print(data.shape)
m, n = data.shape

# We want to randomly shuffle the data in order to split the data. There will be a subset of mnist_train.csv for development (cross validation) and a set for training our model
# We also want to transpose the data so that each columns becomes a single image (28x28) rather than each row
np.random.shuffle(data)

# Captures the first 1000 rows then transposes it so each column is a number
data_dev = data[0:1000].T
Y_dev = data_dev[0] # This captures the first row which is the labels
X_dev = data_dev[1:n] # This captures the pixel data - everything apart from the first row
#print(Y_dev)

# Captures the rest of the data after the first 1000 rows then transposes it so each column is a number
data_train = data[1000:m].T
Y_train = data_train[0] # This captures the first row which is the labels
X_train = data_train[1:n] # This captures the pixel data - everything apart from the first row
#print(Y_train)
#print(Y_dev[:, 0].shape)

# Create a function to initialize parameters
def inti_params():
    w1 = np.random.randn(10, 784) # np.random.randn creates a matrices of specified size with random numbers between -0.5 and 0.5
    b1 = np.random.randn(10, 1)
    w2 = np.random.randn(10, 784)
    b2 = np.random.randn(10, 1)
    return w1 b1 w2 b2

# activation function for hidden layer
# activation function introduce non-linearity
def sigmoid(z):
    return 1/(1 + np.exp(z))

# activation function for output
# if this was not present the output would be a linear function of the output (can't get anything meaningful out of large complex datasets)
def softmax(z):
    return np.exp(z) / np.sum(exp(z))

# Forward propogation - moves from input to output (single forward direction)
def forward_prop(w1, b1, w2, b2, X):
    z1 = w1.dot(X) + b1
    a1 = sigmoid(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1 a1 z2 a2
