import numpy as np

city1 = np.random.randint(-5, 40, size=(7, 24))
city2 = np.random.randint(-5, 40, size=(7, 24))
avg = (city1 + city2) / 2
print("City 1:\n ", city1)
print("City 2:\n ", city2)
print("Average Temperatures:\n", avg)
