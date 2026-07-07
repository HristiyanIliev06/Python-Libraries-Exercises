import numpy as np

n_samples = 50
n_features = 3

# 1. Feature matrix: float32 zeros
X = np.zeros((n_samples, n_features), dtype=np.float32)

# 2. Alternating 0/1 integer labels
labels = np.arange(n_samples) % 2

# 3. Placeholder matching X's shape and dtype
placeholder = np.full(X.shape, -1, dtype=np.float32)

# 4. Verify sample count agreement
assert labels.shape[0] == X.shape[0], "Sample count mismatch"

# 5. Downcast X to float16
X_small = X.astype(np.float16)