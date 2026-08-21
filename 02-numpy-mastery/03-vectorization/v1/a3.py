import numpy as np

values = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
results = np.round(np.exp(values) - np.log(values + 1), 2)
print(results)
