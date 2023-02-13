import torch
from torch import nn
import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
#video timestamp: 10:36:55

device = "cuda" if torch.cuda.is_available() else "cpu"
n_samples = 1000
X, y = make_circles(n_samples, noise=0.03, random_state=42)

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class CircleModelV0(nn.Module):
  def __init__(self):
    super().__init__()
    #takes in 2 features based off of our train data shape, "[(800, 2)]" and outputs 5 features
    self.layer1 = nn.Linear(in_features=2, out_features=5)
    
    #in features has to  match the previous layers' out features or a shape error will occur i.e. 5 from layer one, 5 into layer 2, out features here represents the final output i.e. 1
    self.layer2 = nn.Linear(in_features=5, out_features=1)
    
  def forward(self, x):
    return self.layer2(self.layer1(x))
  
model_0 = CircleModelV0().to(device) 

#this way of writing the model is the same as the class above, a good way to step through a nn if you are going step by step. Likely not as efficient when building more complex nn
#model_0 = nn.Sequential(nn.Linear(in_features=2, out_features=5),nn.Linear(in_features=5, out_features=1)).to(device)

with torch.inference_mode():
  untrained_predictions = model_0(X_test.to(device))
print(f"Predictions length: {len(untrained_predictions)} | Shape: {untrained_predictions.shape}")

loss_fn = nn.BCEWithLogitsLoss()# sigmoid activation function built in
opt = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

#calculating for accuracy
def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct/len(y_pred))*100
  return acc

model_0.eval()
with torch.inference_mode():
  y_logits = model_0(X_test.to(device))[:5]

y_prediction_probs = torch.sigmoid(y_logits)
y_predictions = torch.round(y_prediction_probs)
y_prediction_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))
print(torch.eq(y_predictions.squeeze(), y_prediction_labels.squeeze()))
y_predictions.squeeze()

torch.manual_seed(42)
epochs = 100
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
  model_0.train()
  y_logtis = model_0(X_train).squeeze()
  y_preds = torch.round(torch.sigmoid(y_logits))
  loss = loss_fn(y_logits, y_train)
  acc = accuracy_fn(y_true=y_train, y_pred=y_preds)
  opt.zero_grad()
  loss.backward()
  opt.step()