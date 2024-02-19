import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        num_samples, num_features = X.shape
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for epoch in range(self.epochs):
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)

            gradient_weights = np.dot(X.T, (predictions - y)) / num_samples
            gradient_bias = np.sum(predictions - y) / num_samples

            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias
        print(z)

    def predict(self, X):

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        z = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(z)
        return (predictions >= 0.5).astype(int)


iris = load_iris()
X, y = iris.data, (iris.target == 2).astype(int)  # Binary classification, class 2 vs. rest

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize and fit the logistic regression model for petal length/width
logistic_regression_petal_model = LogisticRegression(learning_rate=0.01, epochs=100)
logistic_regression_petal_model.fit(X_train[:, 2], y_train)

# Initialize and fit the logistic regression model for sepal length/width
logistic_regression_sepal_model = LogisticRegression(learning_rate=0.01, epochs=100)
logistic_regression_sepal_model.fit(X_train[:, :2], y_train)

# Initialize and fit the logistic regression model for all features
logistic_regression_all = LogisticRegression(learning_rate=0.01, epochs=100)
logistic_regression_all.fit(X_train, y_train)

# Make predictions on the test set for each variant
y_pred_petal = logistic_regression_petal_model.predict(X_test[:, 2])
y_pred_sepal = logistic_regression_sepal_model.predict(X_test[:, :2])
y_pred_all = logistic_regression_all.predict(X_test)