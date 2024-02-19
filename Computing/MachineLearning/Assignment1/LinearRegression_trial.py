import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, lr=0.001, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Linear Regression using Gradient Descent.

        Parameters:
        -----------
        lr : float
            Learning rate.
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.lr = lr
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None
        self.batch_loss = []
        self.validation_loss = []

    def fit(self, X, y, X_val=None, y_val=None):
        """Fit a linear model.

        Parameters:
        -----------
        X : numpy.ndarray
            The input training data.
        y : numpy.ndarray
            The target training data.
        X_val : numpy.ndarray, optional
            The input validation data.
        y_val : numpy.ndarray, optional
            The target validation data.
        """
        # Initialize weights and bias based on the shape of X
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient descent
        prev_val_loss = np.inf
        no_improvement_count = 0
        for epoch in range(self.max_epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                # Compute predictions
                y_pred = np.dot(X_batch, self.weights) + self.bias

                # Compute gradients
                dw = (1 / self.batch_size) * np.dot(X_batch.T, (y_pred - y_batch)) + (self.regularization / self.batch_size) * self.weights
                db = (1 / self.batch_size) * np.sum(y_pred - y_batch)

                # Update weights and bias
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

                # Compute MSE loss on batch
                batch_loss = np.mean((y_pred - y_batch) ** 2)
                self.batch_loss.append(batch_loss)

            # Compute validation loss
            if X_val is not None and y_val is not None:
                y_val_pred = np.dot(X_val, self.weights) + self.bias
                val_loss = np.mean((y_val_pred - y_val) ** 2)
                self.validation_loss.append(val_loss)

                # Early stopping: check if validation loss increases for consecutive steps
                if len(self.validation_loss) > self.patience:
                    if all(val_loss > self.validation_loss[-i] for i in range(1, self.patience + 1)):
                        print(f"Stopping early at epoch {epoch}")
                        break

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Training Loss: {batch_loss}")

    def predict(self, X):
        """Predict using the linear model.

        Parameters:
        -----------
        X : numpy.ndarray
            The input data.

        Returns:
        --------
        numpy.ndarray
            The predicted values.
        """
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        """Evaluate the linear model using the mean squared error.

        Parameters:
        -----------
        X : numpy.ndarray
            The input data.
        y : numpy.ndarray
            The target data.

        Returns:
        --------
        float
            The mean squared error.
        """
        y_pred = self.predict(X)
        return np.mean((y_pred - y) ** 2)
