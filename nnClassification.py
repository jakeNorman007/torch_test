import sklearn
from sklearn.datasets import make_circles
'''
video timestamp: 
'''
n_samples = 1000
X, y = make_circles(n_samples, noise=0.03, random_state=42)
print(X[:5])
print(y[:5])