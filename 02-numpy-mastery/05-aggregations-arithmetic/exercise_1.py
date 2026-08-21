import numpy as np

np.random.seed(1)
X = np.random.randn(8, 5)

mean = X.mean(axis = 0)
std = X.std(axis = 0)
min = X.min(axis = 0)
max = X.max(axis = 0)


print(mean)
print(std)
print(min)
print(max)

row_sums = X.sum(axis=1, keepdims=True)   # shape (8, 1) — one sum per row
X_normalised = X / row_sums               # (8,5) / (8,1) → (8,5)
print(X_normalised.sum(axis=1))   # [1., 1., 1., 1., 1., 1., 1., 1.]


# result_list = [mean, std, min, max]
# result = [f"{i+1}#\n{result_list[i]}" for i in range(4)]
# print(*result, sep = '\n')