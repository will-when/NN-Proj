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
    w2 = np.random.randn(10, 10) 
    b2 = np.random.randn(10, 1) 
    return w1, b1, w2, b2

# activation function for hidden layer
# activation function introduce non-linearity
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# activation function for output
# if this was not present the output would be a linear function of the output (can't get anything meaningful out of large complex datasets)
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

# Forward propogation - moves from input to output (single forward direction)
def forward_prop(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1
    a1 = sigmoid(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

# Calculate the derivative of the sigmoid function for use in the back propogation task
def deriv_sigmoid(z):
    return z * (1 - z)

# One hot coding is the conversion of categorical information into a format that may be fed into machine learning algorithms to improve prediction accuracy
# This takes a a category and represent it in a one-hot way which is powerful when dealing with data that is non-numerical especially.
def one_hot(y):
    one_hot_Y = np.zeros((y.size), y.max() + 1) # Creates an array of zeros. the size is defined by y.size (which is m - number of rows) and y.max + 1 assumes 9 classes and add 1 to get 10 which is the desired number of outputs (1-10)
    one_hot_Y[np.arange(y.size)] = 1 # index through the one_hot_Y using arrays from 0 to m (y.size) and y is specifying the column it accesses (so essentiall going to each row and accessing the label column and seting it to 1)
    one_hot_Y = one_hot_Y.T # transposing the array so all labels are in a row 
    return one_hot_Y

# Backwards propogation - essentially allows model to determine the error in previous iteration so it can improve accuracy of classificaion
def backward_prop(z1, a1, z2, a2, w1, w2, x, y):
    one_hot_Y = one_hot(y)
    dz2 = a2 - one_hot_Y
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2, axis=1, keepdims=True)
    dz1 = w2.T.dot(dz2) * sigmoid(z1)
    dw1 = 1 / m * dz1.dot(x.T)
    db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)
    return dw1, db1, dw2, db2

# Now using the back propogation we are going to update the weighting and also bias
def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1    
    w2 = w2 - alpha * dw2  
    b2 = b2 - alpha * db2    
    return w1, b1, w2, b2

# Function which gets the predictions from our model
def get_predictions(a2):
    return np.argmax(a2, 0)

# this functions determines the accuracy of our predicitons
def get_accuracy(predictions, y):
    print(predictions, y)
    return np.sum(predictions == y) / y.size # sum of the predictions over the number of examples there are

# Now we can train the neural network using gradient descent
def gradient_descent(x, y, iterations, alpha):
    w1, b1, w2, b2 = inti_params()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = backward_prop(z1, a1, z2, a2, w1, w2, x, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if (i % 100 == 0):
            print("Iteration: ", i)
            predictions = get_predictions(a2)
            print("Accuracy: ", get_accuracy(predictions, y))
    return w1, b1, w2, b2

w1, b1, w2, b2 = gradient_descent(X_train, Y_train, 1000, 0.1)