import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('icbhi_dataset.csv')
vec_A = df.iloc[0, 0:13].values
vec_B = df.iloc[1, 0:13].values

def minkowski(a, b, p):
    sum_pow = sum(abs(x - y)**p for x, y in zip(a, b))
    return sum_pow**(1/p)

p_values = range(1, 11)
distances = [minkowski(vec_A, vec_B, p) for p in p_values]

plt.figure(figsize=(10, 6))
plt.plot(p_values, distances, marker='o', linestyle='-', color='purple')
plt.title('Minkowski Distance vs p')
plt.xlabel('p')
plt.ylabel('Distance')
plt.grid(True)
plt.show()