import numpy as np

w = 2.5
b= -1.2
x = np.array([np.arange(1, 5, 0.5)])
y = w * x + b
print(np.round((y).mean(), 3))