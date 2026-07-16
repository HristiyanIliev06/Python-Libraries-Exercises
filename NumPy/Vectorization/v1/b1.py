import numpy as np
#[0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0]
a  = np.random.rand(1, 1000)

conditions = [a < 0.25, (a >= 0.25) & (a < 0.5), (a >= 0.5) & (a < 0.75), a >= 0.75]
choices = [0, 1, 2, 3]
x = np.select(conditions, choices)

count1 = np.sum(x == 0)
count2 = np.sum(x == 1) 
count3 = np.sum(x == 2)
count4 = np.sum(x == 3)
total = (count1 + count2 + count3 + count4)/100

print(f"Percentage of values in [0, 0.25): {count1/total}%")
print(f"Percentage of values in [0.25, 0.5): {count2/total}%")
print(f"Percentage of values in [0.5, 0.75): {count3/total}%")
print(f"Percentage of values in [0.75, 1.0): {count4/total}%")