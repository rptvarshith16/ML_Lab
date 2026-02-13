import numpy as np
import pandas as pd

# Load data
df = pd.read_csv('icbhi_dataset.csv')

vec_A = df.iloc[0, 0:13].values
vec_B = df.iloc[1, 0:13].values

def custom_dot_product(a, b):
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

def custom_euclidean_norm(a):
    sq_sum = 0
    for x in a:
        sq_sum += x**2
    return sq_sum**0.5

dot_custom = custom_dot_product(vec_A, vec_B)
norm_A_custom = custom_euclidean_norm(vec_A)

dot_numpy = np.dot(vec_A, vec_B)
norm_A_numpy = np.linalg.norm(vec_A)

print("Dot Product: Custom =", dot_custom, "| Numpy =", dot_numpy)
print("Euclidean Norm (A): Custom =", norm_A_custom, "| Numpy =", norm_A_numpy)