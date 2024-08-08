from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from Perceptron import  Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

iris = load_iris() 
X = iris.data[:, (0, 1)] 
y = (iris.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
perceptron = Perceptron(epochs=10)

perceptron.fit(X_train, y_train)
pred = perceptron.predict(X_test)

print(f"Accuracy : {accuracy_score(pred, y_test)}")
report = classification_report(pred, y_test, digits=2)
