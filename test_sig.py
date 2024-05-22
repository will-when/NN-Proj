# This has been created by William Peters on the 14/05/2024
# Use at your own risk and ensure you do your own due dilligence

# Import neccessary modules for the project

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Convert read in data from csv using pandas module
data = pd.read_csv('mnist_train.csv')
#print(data.head(10)) - Prints the first 10 rows of the csv that is being read in

# Convert data to a numpy array - this will make it easy to do matrix multiplication when it comes to forwards and bacwards propogation
data = np.array(data)
#print(data.shape)


# Determine number of rows and also columns
#print(data.shape)
m, n = data.shape # this assigns the values m = 60000 rows and n = 785 columns

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
X_train = data_train[1:n] # This captures the pixel data - everything apart from the first row (labels)
#X_train = X_train.astype('float64')
#Y_train = Y_train.astype('float64')
#print(X_train.shape)
#print(X_train.dtype)
#print(Y_train.dtype)
#print(Y_dev[:, 0].shape)

# Create a function to initialize parameters

w1 = np.random.randn(10, 784) # np.random.randn creates a matrices of specified size with random numbers between -0.5 and 0.5
b1 = np.random.randn(10, 1) 
w2 = np.random.randn(10, 10) 
b2 = np.random.randn(10, 1) 
#print(w1.shape)

# Forward propogation - moves from input to output (single forward direction)
z1 = w1.dot(X_train) + b1
#print(z1.shape)
a1 = z1 * (1 - z1)
#print(a1.shape)
z2 = w2.dot(a1) + b2
#print(z2.shape)
a2 = np.exp(z2) / np.sum(np.exp(z2))
#print(a2.size)

# One hot coding is the conversion of categorical information into a format that may be fed into machine learning algorithms to improve prediction accuracy
# This takes a a category and represent it in a one-hot way which is powerful when dealing with data that is non-numerical especially.
one_hot_Y = np.zeros((Y_train.size, Y_train.max() + 1)) # Creates an array of zeros. the size is defined by y.size (which is m - number of rows) and y.max + 1 assumes 9 classes and add 1 to get 10 which is the desired number of outputs (1-10)
#print(Y_train.max()+1)
#print(one_hot_Y.size)
one_hot_Y[np.arange(Y_train.size), Y_train] = 1 # index through the one_hot_Y using arrays from 0 to m (y.size) and y is specifying the column it accesses (so essentiall going to each row and accessing the label column and seting it to 1)
#print(np.arange(Y_train.size))
#print(one_hot_Y)
one_hot_Y = one_hot_Y.T # transposing the array so all labels are in a row
#print(one_hot_Y) 

# Backwards propogation - essentially allows model to determine the error in previous iteration so it can improve accuracy of classificaion
dz2 = a2 - one_hot_Y
#print(dz2.shape)
dw2 = 1 / m * dz2.dot(a1.T)
#print(dw2.shape)
db2 = 1 / m * np.sum(dz2, axis=1, keepdims=True)
#print(db2.shape)
dz1 = w2.T.dot(dz2) * (z1 * (1 - z1))
#print(dz1.shape)
dw1 = 1 / m * dz1.dot(X_train.T)
#print(dw1.shape)
db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)
#print(db1.shape)