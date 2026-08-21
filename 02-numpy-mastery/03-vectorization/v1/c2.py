import numpy as np

data = np.random.randn(500)

mean: float = data.mean()
std: float  = data.std()

# A fault is any reading MORE than 3 std away from the mean (either direction)
threshold = 3 * std
fault_mask = np.abs(data - mean) > threshold

fault_indices = np.where(fault_mask)[0]

median_clean = np.median(data[~fault_mask])   # ~ inverts the mask

data[fault_mask] = median_clean

print(f"Data before cleaning: {data}")
print(f"Mean: {mean}, Std: {std}, Threshold: {threshold}")
print(f"Number of faults detected ({fault_indices}) and replaced: {len(fault_indices)}" )
print(f"Median of clean data: {median_clean}")
print(f"Data after cleaning: {data}")
