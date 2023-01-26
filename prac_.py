import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# print(torch.__version__)

# Tenosrs and Matrix are usually named in CAPS i.e TENSOR, MATRIX

# X = torch.rand(3, 4)

# # print(X.ndim) # 2 in this case. number of dimensions can be determined by the number of indexes in the random tensor we created

# # Creating a random tensor with a simillar shape to an image tensor
# # name = torch.rand(size=(heigh, width, color channels(Red, Green, Blue)))

# IMG_TENSOR = torch.rand(size=(224, 224, 3))
# print(IMG_TENSOR.shape, IMG_TENSOR.ndim)


MATRIX = torch.tensor([[1, 2, 3], [3, 1, 2]])
MATRIX_2 = torch.tensor([[1, 3], [1, 1], [4, 5]])

print(MATRIX @ MATRIX_2)

#2:34:50 - time stamp for the course video