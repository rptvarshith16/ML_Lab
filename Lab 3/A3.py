import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('icbhi_dataset.csv')
feature_data = df['mfcc0']

mean_val = np.mean(feature_data)
var_val = np.var(feature_data)

print(f"Feature: mfcc0 | Mean: {mean_val:.2f} | Variance: {var_val:.2f}")

plt.figure(figsize=(10, 6))
plt.hist(feature_data, bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of mfcc0')
plt.xlabel('mfcc0 Values')
plt.ylabel('Frequency')
plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.legend()
plt.show()