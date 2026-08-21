import numpy as np

np.random.seed(5)
X = np.random.randn(8, 4)

# 1. Per-column z-score normalization
col_means = X.mean(axis=0)     # shape (4,)
col_stds  = X.std(axis=0)      # shape (4,)
X_norm = (X - col_means) / col_stds   

# 2. Per-row normalization by row sum
row_sums = X.sum(axis=1).reshape(-1, 1)       # shape (8, 1)
X_row = X / row_sums           

# 3. Add a bias vector to every sample
bias = np.array([1, 0, -1, 0]) # shape (4,)
out  = X_norm + bias            

# 4. Outer product of two vectors
a = np.array([1, 2, 3])        # shape (3,)
b = np.array([10, 20, 30])     # shape (3,)
outer = a.reshape(-1, 1) * b                  

# 5. Scale each column by a different factor
scales = np.array([2.0, 0.5, 1.0, 3.0])  # shape (4,)
X_scaled = X_norm * scales