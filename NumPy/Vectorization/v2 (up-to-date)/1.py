import numpy as np

np.random.seed(0)
X = np.random.randn(100, 5)

mean = X.mean()
std =  X.std()
z_score_normalization_of_X = (X - mean) / std
print(f"Z-score normalization of X: {z_score_normalization_of_X} with mean: {mean:.6f} & std: {std:.6f}")

X_no_negatives = np.where(X < 0, 0, X)
count_of_clipped_values = np.sum(X < 0)
print(count_of_clipped_values)