import numpy as np

weight_matrix = np.zeros((8, 4), dtype = np.float32)
bias_vector = np.ones((8,), dtype = np.float32)
learning_rate_candidates = np.linspace(1e-4, 1e-1, 6)

print(weight_matrix)
print(bias_vector)
print(learning_rate_candidates)

x = np.arange(24)
X = x.reshape(2, 3, 4)
print(f"Attributes of X:  shape: {X.shape}, ndim: {X.ndim}, dtype: {X.dtype}, and nbytes: {X.nbytes}")
# X.nbytes = 192 = X.size * X.itemsize = 24 * 8
# 8 bytes per element because dtype is int64 (64-bit integer, default on 64-bit systems)