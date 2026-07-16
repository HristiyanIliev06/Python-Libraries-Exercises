import numpy as np

temps_c = np.array([-40, -20, 0, 20, 37, 100])
temps_f = (temps_c * 9/5) + 32

conditions = [temps_f < 32, (temps_f >= 32) & (temps_f < 68), (temps_f >= 68) & (temps_f < 86), temps_f >= 86]
choices = [" freezing", "cold", "comfortable", "hot"]

categories = np.select(conditions, choices)

print("Temperatures in Celsius:", temps_c)
print("Temperatures in Fahrenheit:", temps_f)
print(categories)

