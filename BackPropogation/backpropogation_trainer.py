from BackPropogation import  BackPropogation
import numpy as np


if __name__ == "__main__":
    
    # OR gate dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])  

    backpropogation = BackPropogation(epochs=10)
    backpropogation.fit(X, y)

    test_data = np.array([[0, 1], [0, 0], [0, 0], [1, 0]])
    predictions = backpropogation.predict(test_data)

    print("Predictions :", predictions)
    