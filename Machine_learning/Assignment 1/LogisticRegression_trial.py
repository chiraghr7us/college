import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self, lr=0.01, epoch=100):
        self.weights = None
        self.bias = None
        self.lr = lr
        self.epoch = epoch

    def predict(self, X):

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        z = np.dot(X, self.weights) + self.bias
        pred = self.sigmoid(z)
        return (pred >= 0.5).astype(int)

    def fit(self, X, y):

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        num_of_samples, num_of_features = X.shape
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.epoch):
            z = np.dot(X, self.weights) + self.bias
            pred = self.sigmoid(z)

            dw = np.dot(X.T, (pred - y)) / num_of_samples
            db = np.sum(pred - y) / num_of_samples

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        print(z)


    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


# iris = load_iris()
# X, y = iris.data, (iris.target == 2).astype(int)  # Binary classification, class 2 vs. rest
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
#
# # Initialize and fit the logistic regression model for petal length/width
# logistic_regression_petal_model = LogisticRegression(lr=0.01, epoch=100)
# logistic_regression_petal_model.fit(X_train[:, 2], y_train)
#
# # Initialize and fit the logistic regression model for sepal length/width
# logistic_regression_sepal_model = LogisticRegression(lr=0.01, epoch=100)
# logistic_regression_sepal_model.fit(X_train[:, :2], y_train)
#
# # Initialize and fit the logistic regression model for all features
# logistic_regression_all = LogisticRegression(lr=0.01, epoch=100)
# logistic_regression_all.fit(X_train, y_train)
#
# # Make pred on the test set for each variant
# y_pred_petal = logistic_regression_petal_model.predict(X_test[:, 2])
# y_pred_sepal = logistic_regression_sepal_model.predict(X_test[:, :2])
# y_pred_all = logistic_regression_all.predict(X_test)