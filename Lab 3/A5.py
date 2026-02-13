import numpy as np
import pandas as pd
from scipy.spatial import distance
from A4 import *

df = pd.read_csv('icbhi_dataset.csv')
vec_A = df.iloc[0, 0:13].values
vec_B = df.iloc[1, 0:13].values

dist_custom = minkowski(vec_A, vec_B, 3)

dist_scipy = distance.minkowski(vec_A, vec_B, 3)

print(f"Minkowski Distance Calculated = {dist_custom}")
print(f"Minkowski Distance scipy = {dist_scipy}")