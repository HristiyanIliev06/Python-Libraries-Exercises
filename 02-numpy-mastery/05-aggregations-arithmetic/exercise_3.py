import numpy as np

def batch_normalize(X: np.ndarray):
    """
    X has shape (batch_size, num_features).
    Normalize each feature (column) to zero mean, unit variance across the batch.
    """
    mean = X.mean(axis=1, keepdims = True)
    std = X.std(axis=0)
    
    normalized = (X - mean) / std
    
    return normalized

def summarize_features(X: np.ndarray):
    """
    Return the index of the feature (column) with the highest variance,
    and the index of the sample (row) with the highest total activation (sum across features).
    """
    highest_var_feature = np.argmax(X.std(axis=0))
    max_val = X.max() # unclear use
    highest_activation_sample = np.argmax(np.sum(X, axis=1))
    
    return highest_var_feature, highest_activation_sample

X = np.dot(np.random.randn(100, 1), np.random.randn(1, 20)) + np.random.randn(100, 20) * 0.1
X = np.random.randn(50, 8) * np.array([1, 5, 0.1, 3, 10, 2, 0.5, 4]) + np.array([0, 10, -5, 0, 100, -20, 3, 7])
Ẍ = np.copy(X)

print(batch_normalize(X))
# row_sums = X.sum(axis=1, keepdims=True)   # shape (8, 1) — one sum per row
# X_normalised = X / row_sums               # (8,5) / (8,1) → (8,5)
# print(X_normalised.sum(axis=1)) 
print('\n')
print(summarize_features(X))