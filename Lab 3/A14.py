import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("icbhi_dataset.csv")

X = df.iloc[:, :13].values
y = df["crackle"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

def matrix_model(X_train, y_train, X_test):
    X_train = np.c_[np.ones(len(X_train)), X_train]
    X_test = np.c_[np.ones(len(X_test)), X_test]

    w = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
    y_pred = X_test @ w

    return (y_pred >= 0.5).astype(int)

y_pred_matrix = matrix_model(X_train, y_train, X_test)
acc_matrix = accuracy_score(y_test, y_pred_matrix)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)

print("Matrix Inversion Accuracy:", round(acc_matrix, 4))
print("kNN Accuracy             :", round(acc_knn, 4))

if acc_knn > acc_matrix:
    print("Result: kNN performs better")
else:
    print("Result: Matrix Inversion performs better")
