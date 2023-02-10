import torch
from torch import nn
import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

'''
video timestamp: 
'''
device = "cuda" if torch.cuda.is_available() else "cpu"

n_samples = 1000
X, y = make_circles(n_samples, noise=0.03, random_state=42)

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class CircleModelV0(nn.Module):
  def __init__(self):
    super().__init__()
    # takes in 2 features based off of our train data shape, "[(800, 2)]" and outputs 5 features
    self.layer1 = nn.Linear(in_features=2, out_features=5)
    
    # in features has to  match the previous layers' out features or a shape error will occur i.e. 5 from layer one, 5 into layer 2, out features here represents the final output i.e. 1
    self.layer2 = nn.Linear(in_features=5, out_features=1)
    
  def forward(self, x):
    return self.layer2(self.layer1(x))

model_0 = CircleModelV0().to(device) 

# this way of writing the model is the same as the class above, a good way to step through a nn if you are going step by step. Likely not as efficient when building more complex nn
model_0 = nn.Sequential(nn.Linear(in_features=2, out_features=5),nn.Linear(in_features=5, out_features=1)).to(device)