import numpy as np

x = np.array([2.0, 5.0, 9.0, 16.0, 25.0])
result = np.round(x**2 - np.sqrt(x), 2)
print(result)