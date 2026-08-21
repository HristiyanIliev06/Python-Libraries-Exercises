import numpy as np

np.random.seed(3)
X = np.random.randn(6, 4)
# print(X)

column_mean = X.mean(axis = 0)
result = X - column_mean

print(result)

L2_X_row = np.sqrt(np.sum(X**2, axis = 1))
result_norm = X / L2_X_row.reshape(-1, 1)
row_norms_after = np.sqrt(np.sum(result_norm**2, axis=1))
print(row_norms_after)   # should all be ~1.0