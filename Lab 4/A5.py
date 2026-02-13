import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("icbhi_dataset.csv")

df["class"] = df["crackle"] + df["wheeze"]

X = df[["mfcc0", "mfcc1"]].values
y = df["class"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

test_points = np.c_[xx.ravel(), yy.ravel()]

k_values = [1, 3, 5, 7, 15]

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    predictions = knn.predict(test_points)
    plt.figure()
    plt.scatter(test_points[:,0], test_points[:,1], c=predictions, s=5)
    plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k')
    plt.title(f"k = {k}")
    plt.show()
