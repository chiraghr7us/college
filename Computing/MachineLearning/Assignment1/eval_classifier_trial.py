import numpy as np
from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LogisticRegression_trial import LogisticRegression

bc = datasets.load_breast_cancer()
iris = load_iris()
# X, y = bc.data[:,0], bc.target
X, y = iris.data, iris.target
print("data shape:",X.shape,y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = LogisticRegression()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(y_pred)
print(y_test)
# plot_decision_regions(X_test, y_test, clf=clf, legend=2)
def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)

acc = accuracy(y_pred, y_test)
print(acc)

