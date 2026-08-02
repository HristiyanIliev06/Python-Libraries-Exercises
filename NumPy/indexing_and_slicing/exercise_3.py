import numpy as np

data = np.arange(1, 49).reshape(6, 8)

# 1. Get the first column
col = data[: ,0]

# 2. Get rows 2 through 4 (inclusive)
rows = data[2:5] # last index with slicing always excluded

# 3. Get the top-right 2x4 submatrix
submatrix = data[0:2, -4:]

# 4. Reverse columns
reversed_cols = data[:, ::-1]

# 5. Extract a copy of the last 3 rows
last_rows = data[-3:].copy

# 6. Confirm it's a copy
print(np.shares_memory(data, last_rows))

print(data)
print(submatrix)