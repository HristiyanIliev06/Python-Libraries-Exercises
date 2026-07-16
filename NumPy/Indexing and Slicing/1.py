import numpy as np

X = np.arange(1, 37).reshape(6, 6)

column_3 = X[:, 2]

matrix_3x3_bot_r = X[3:6, 3:6] # X[-3:, -3:] is more robust in production

reversed_X = X[::-1]



print(X)
print()
print(column_3)
print()
print(matrix_3x3_bot_r)
print()
print(reversed_X)
print()

x = X[2:5, 1:4]
x[0, 1] = 99

print(np.shares_memory(X, x))
print(X)


