import numpy as np

np.random.seed(7)
X = np.random.randn(200, 6)

# Simulate some outliers and missing sentinels
X[::15, 2] = 99.0     # outlier values in feature 2
X[::20, 4] = -999.0   # sentinel "missing" values in feature 4

col2 = X[:, 2]
clean_mean = col2[col2 <= 10.0].mean()
X[:, 2] = np.where(col2 > 10.0, clean_mean, col2)

col4 = X[:, 4]
valid_median = np.median(col4[col4 != -999.0])
X[:, 4] = np.where(col4 == -999.0, valid_median, col4)

final_scaled_X = (X - X.min()) / (X.max() - X.min())

labels = np.array(['cat', 'dog', 'bird', 'dog', 'cat', 'bird', 'cat'])
conditions = [labels == 'cat', labels == 'dog', labels == 'bird']
choices = [0, 1, 2]
encoded_matrix = np.select(conditions, choices)

print(final_scaled_X)
print(final_scaled_X.shape)
print(final_scaled_X.dtype)
print(final_scaled_X.min())
print(final_scaled_X.max())