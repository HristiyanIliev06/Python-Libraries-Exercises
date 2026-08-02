import numpy as np

np.random.seed(0)
X = np.random.randn(50, 8)   # 50 samples, 8 features

X_feature_mean = X.mean(axis = 0)
X_feature_variance = X.var(axis = 0)

epsilon = 1e-8
X_norm = (X - X_feature_mean) / np.sqrt(X_feature_variance + epsilon)

print(X_norm.shape())
print(X_feature_mean)
print(X_feature_variance)

gamma = np.ones(8)
beta = np.zeros(8)
out = gamma * X_norm + beta