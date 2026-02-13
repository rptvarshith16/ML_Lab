import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("icbhi_dataset.csv")

df["class"] = df["crackle"] + df["wheeze"]

X = df[["mfcc0", "mfcc1"]].values
y = df["class"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

param_grid = {'n_neighbors': range(1, 31)}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X, y)

print("Best k:", grid.best_params_)
print("Best accuracy:", grid.best_score_)