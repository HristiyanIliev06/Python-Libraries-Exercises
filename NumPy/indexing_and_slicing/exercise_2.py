import numpy as np

np.random.seed(42)
X = np.random.randn(120, 8)
y = (np.random.rand(120) > 0.5).astype(int)

X_train = X[:96]
X_test  = X[96:]
y_train = y[:96]
y_test  = y[96:]

# Fancy indexing — always a copy, shape (96, 4)
X_train_selected = X_train[:, [0, 3, 5, 7]]

# Every other sample from the test set, shape (12, 8)
X_test_sampled = X_test[::2]

mean_train = X_train[:, 2].mean()
mean_test  = X_test[:, 2].mean()
print(mean_train)
print(mean_test)

# X_train_selected does not share memory with X because fancy indexing
# (indexing with a list) always returns a copy — the selected columns
# are not contiguous in memory, so NumPy cannot create a view.
print(np.shares_memory(X, X_train_selected))  # False

