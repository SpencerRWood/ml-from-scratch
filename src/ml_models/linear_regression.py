import numpy as np

class LinearRegression: 
    def __init__(self, lr = 0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        #init weights and biases as 0
        #need one weight for each feature of X
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        #bias is initialized as 0 because it 
        self.bias = 0

        for _ in range(self.n_iters):

        #to get y_pred, calculated the dot product 
        # between X and weights
        #note: dot product takes two equal length sequences
        # of numbers (vectors) and returns a single number
        # Numpy includes the summation (sigma)
            y_pred = np.dot(X, self.weights) + self.bias
            #gradient descent
            dw = (1/n_samples) * np.dot(X.T,(y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weight = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)