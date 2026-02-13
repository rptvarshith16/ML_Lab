import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('icbhi_dataset.csv')
X = df.iloc[:, 0:13].values
y = df['crackle'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

print("Training Performance")
print("Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))

print("Testing Performance")
print("Confusion Matrix:", confusion_matrix(y_test, y_test_pred))

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"Train Accuracy: {train_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")

if train_acc > test_acc + 0.15:
    print("Inference: Model is Overfitting (High variance)")
elif train_acc < 0.6:
    print("Inference: Model is Underfitting (High bias)")
else:
    print("Inference: Regular Fit (Good generalization)")