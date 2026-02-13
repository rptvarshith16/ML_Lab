import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('icbhi_dataset.csv')

X = df.iloc[:, 0:13]
y = df['crackle']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Original Shape:", X.shape)
print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)