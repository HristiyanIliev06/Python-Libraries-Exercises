import numpy as np

# 1. 1000 float64 values
data = np.random.rand(1000)

# 2. Reshape to 200 samples × 5 features
X = data.reshape(200, 5)

# 3. Sentinel placeholder, same shape
placeholder = np.full((200, 5), -999.0)

# 4. Integer labels: 100 × class 0, 100 × class 1
labels = np.concatenate([np.zeros(100, dtype=int), np.ones(100, dtype=int)])
print(labels)

# 5. Memory footprint comparison
X_f32 = X.astype(np.float32)
print(f"float64 nbytes: {X.nbytes}")       # 8000
print(f"float32 nbytes: {X_f32.nbytes}")   # 4000
print(f"ratio: {X.nbytes / X_f32.nbytes}") # 2.0
