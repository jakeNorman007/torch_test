import torch
from torch import nn # contains all the PT building blocks for neural networks (nn)
import matplotlib.pyplot as plt # for visualization

'''
01 - basic Pytorch workflow

- data, prepare and load data
- build a simple model
- fitting the model to the data we load (training)
- make predictions and evaluate them (Inference)
- saving and loading a model
- finally, put it all together

-- video time stamp 

-- notes, research links etc. --

- linear regression formula (Y = a + bX), where X is the explanatory variable and Y is the dependent variable. The slope of the line (b), the intercept of the line (a) -> the value of y when x = 0

'''

# using linear regression to make a straight line using known data

# known parameters
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1) #unsqueeze adds another dimension so that the tensors are not in the 0th dimension but the 1st
y = weight * X + bias

X[:10], y[:10]

# splitting data into training and test sets. Very important to know
# createing a test / train split

train_split = int(0.8 * len(X)) # creates a training split of 80/ 20 in our data. Which is the upper bounds of typical splits.
X_train, y_train = X[:train_split], y[:train_split] # trains on 80% of the data
X_test, y_test = X[train_split:], y[train_split:] # tests on 20% of the data

print(len(X_train), len(y_train), len(X_test), len(y_test))