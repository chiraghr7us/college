import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Classification import LogisticRegression  # Assuming you have implemented the Logistic Regression class
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X, y = iris.data,(iris.target == 2).astype(int)   # Binary classification, class 2 vs. rest

# Standardize the input features
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.1, random_state=42, stratify=y)

# Initialize and fit the logistic regression model
logistic_regression_petal_model = LogisticRegression(learning_rate=0.01, epochs=100)
#logistic_regression_petal_model.fit(X_train[:, 2:4], y_train)
logistic_regression_petal_model.fit(X_train[:, 2], y_train)
print("\nActual values",y_test)

# Make predictions on the test set
y_pred_petal = logistic_regression_petal_model.predict(X_test[:, 2])
print("\npredicted values", y_pred_petal)
# Evaluate accuracy
accuracy = np.mean(y_pred_petal == y_test)
print(f"\nAccuracy (Petal): {accuracy}")

plt.figure(figsize=(12, 5))
# Petal Length/Width
# plot_decision_regions(X_test[:, 2:], y_test, clf=logistic_regression_petal_model, legend=2)
# plt.xlabel('Petal Length (Standardized)')
# plt.ylabel('Petal Width (Standardized)')
# plt.title('Logistic Regression Decision Regions (Petal)')
# plt.show()

#plot_decision_regions(X_test[:, 2:4], y_test, clf=logistic_regression_petal_model, legend=2)
plot_decision_regions(X_test[:, 2].reshape(-1, 1), y_test, clf=logistic_regression_petal_model, legend=2)
plt.xlabel('Petal Length (Standardized)')
plt.ylabel('Petal Width (Standardized)')
plt.title('Logistic Regression Decision Regions (Petal)')
plt.show()