import numpy as np
from tqdm import tqdm


class BackPropogation:
    def __init__(self,learning_rate=0.01, epochs=100,activation_function='step'):
        self.bias = 0
        self.learning_rate = learning_rate
        self.max_epochs = epochs
        self.activation_function = activation_function


    def activate(self, x):
        if self.activation_function == 'step':
            return 1 if x >= 0 else 0
        elif self.activation_function == 'sigmoid':
            return 1 if (1 / (1 + np.exp(-x)))>=0.5 else 0
        elif self.activation_function == 'relu':
            return 1 if max(0,x)>=0.5 else 0

    def fit(self, X, y):
        error_sum=0
        n_features = X.shape[1]
        self.weights = np.zeros((n_features))
        for epoch in tqdm(range(self.max_epochs)):
            for i in range(len(X)):
                inputs = X[i]
                target = y[i]
                weighted_sum = np.dot(inputs, self.weights) + self.bias
                prediction = self.activate(weighted_sum)
                
                # Calculating loss and updating weights.
                error = target - prediction
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error
                
            print(f"Updated Weights after epoch {epoch} with {self.weights}")
        print("Training Completed")

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            inputs = X[i]
            weighted_sum = np.dot(inputs, self.weights) + self.bias
            prediction = self.activate(weighted_sum)
            predictions.append(prediction)
        return predictions


    
    
    
 