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

class SimpleKNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test):
        predictions = []

        for test_point in X_test:
            distances = np.sqrt(np.sum((self.X - test_point) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y[k_indices]
            values, counts = np.unique(k_labels, return_counts=True)
            predictions.append(values[np.argmax(counts)])

        return np.array(predictions)

k = 3

custom_knn = SimpleKNN(k)
custom_knn.fit(X_train, y_train)
custom_predictions = custom_knn.predict(X_test)

sklearn_knn = KNeighborsClassifier(n_neighbors=k)
sklearn_knn.fit(X_train, y_train)
sklearn_predictions = sklearn_knn.predict(X_test)

print("Custom kNN Accuracy:", accuracy_score(y_test, custom_predictions))
print("Sklearn kNN Accuracy:", accuracy_score(y_test, sklearn_predictions))
