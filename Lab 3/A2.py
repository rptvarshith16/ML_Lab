import numpy as np
import pandas as pd

df = pd.read_csv('icbhi_dataset.csv')

def calculate_class_stats(data_matrix):
    mean_vec = np.mean(data_matrix, axis=0)
    std_vec = np.std(data_matrix, axis=0)
    return mean_vec, std_vec

class_0_data = df[df['crackle'] == 0].iloc[:, 0:13].values
class_1_data = df[df['crackle'] == 1].iloc[:, 0:13].values

centroid_0, spread_0 = calculate_class_stats(class_0_data)
centroid_1, spread_1 = calculate_class_stats(class_1_data)

interclass_distance = np.linalg.norm(centroid_0 - centroid_1)

print("Class 0 Centroid:", centroid_0[:3])
print("Class 1 Centroid:", centroid_1[:3])
print("Interclass Distance:", interclass_distance)