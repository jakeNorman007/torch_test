import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# print(torch.__version__)

x = torch.rand(5,3)
y = torch.tensor([7, 7])
print(y.ndim)
print(y.shape)

MATRIX = torch.tensor([[7, 7], [9, 10]])

print(MATRIX)
print(MATRIX.ndim)
print(MATRIX.shape)

TENSOR = torch.tensor([[[4, 6, 2], [5, 9, 1], [9, 8, 7]]])

print(TENSOR)
print(TENSOR.ndim)
print(TENSOR.shape)