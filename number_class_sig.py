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

# Captures the rest of the data after the first 1000 rows then transposes it so each column is a number
data_train = data[1000:m].T
Y_train = data_train[0] # This captures the first row which is the labels
X_train = data_train[1:n] # This captures the pixel data - everything apart from the first row
