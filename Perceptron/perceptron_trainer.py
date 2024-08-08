from Perceptron import  Perceptron
import numpy as np


if __name__ == "__main__":
    
    # OR gate dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])  

    perceptron = Perceptron(epochs=10)
    perceptron.fit(X, y)

    test_data = np.array([[0, 1], [1, 0], [0, 0], [1, 0]])
    predictions = perceptron.predict(test_data)

    print("Predictions :", predictions)