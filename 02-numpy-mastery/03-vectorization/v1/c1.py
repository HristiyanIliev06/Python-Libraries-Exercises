import numpy as np

random_integers = np.random.randint(10, 100, size=(5, 4))

global_max = np.max(random_integers)
global_min = np.min(random_integers)

column_max = np.max(random_integers, axis=0)
column_min = np.min(random_integers, axis=0)

print("Random Integers:\n", random_integers)
print("Global Maximum:", global_max, )
print("Global Minimum:", global_min)
print("Column Maximum:", column_max)
print("Column Minimum:", column_min)