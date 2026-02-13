import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('icbhi_dataset.csv')
neigh = KNeighborsClassifier(n_neighbors = 3)
X = df.iloc[:, 0:13]
y = df['crackle']

print(neigh.fit(X,y))
print(neigh.score(X,y))
print(neigh.predict(X))
