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

-- video time stamp 4:52:49

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

def plot_predictions(train_data=X_train, train_labels=y_train, 
                     test_data=X_test, test_labels=y_test, 
                     predictions=None):
  '''
  Plots the trining data, test data and compares predictions.
  '''
  plt.figure(figsize=(10, 7))
  # training data
  plt.scatter(train_data, train_labels, c='b', s=4, label="Training data")
  
  # test data
  plt.scatter(test_data, test_labels, c='g', s=4, label="Test data")
  if predictions is not None:
    plt.scatter(test_data, predictions, c='r', s=4, label="Predictions")
    
  plt.legend(prop={"size": 14})
  plot_predictions()
  
# First model build (Linear regression model)
  
class LinearRegressionModel(nn.Module): # almost everything in PT inherhits from nn.Module
  def __init__(self):
    super().__init__()
    self.weights, self.bias = nn.parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
    
  #forward method to define computation in the model
  def forward(self, x: torch.Tensor) -> torch.Tensor: # x is the input data, which is tensor, needs to return tensor
    return self.weights * x + self.bias
