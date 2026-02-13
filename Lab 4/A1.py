import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

data = pd.read_csv("icbhi_dataset.csv")

X = data.iloc[:, 0:13] # row 
y = data["crackle"]    

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

k1=confusion_matrix(y_train, y_train_pred)
k2=confusion_matrix(y_test, y_test_pred)

print(k1)
print(k2)

# metrices
#fp,np,tn,tp.
