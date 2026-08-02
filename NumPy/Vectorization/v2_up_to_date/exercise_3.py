import numpy as np

np.random.seed(1)
X = np.random.randn(50, 4)

# 1. Z-score normalize each column (per-column mean and std)
X_norm = (X - X.mean(axis=0)) / X.std(axis=0) # mean and std used to be global (no axis=0)

# 2. ReLU via np.where
relu = np.where(X_norm > 0, X_norm, 0) # missing return value for unmet condition

# 3. Replace values below -2 with -2 (lower clip)
clipped = np.clip(X_norm, -2, np.inf)  # or use None for no upper bound
clipped = np.clip(X_norm, -2, None)    # None means unbounded
clipped = np.where(X_norm < -2, -2, X_norm) # missing argument on the previosly used np.clip(...)

# 4. Label encoding with np.select
labels = np.array(['cat', 'dog', 'bird', 'dog', 'cat'])
conditions = [labels == 'cat', labels == 'dog', labels == 'bird']
encoded = np.select(conditions, [0, 1, 2], default=0)

# 5. Use np.vectorize for fast per-element transformation
X_transformed = np.log(np.abs(X_norm) + 1) * np.sign(X_norm) # vectorize() worked but no performance advantage gained