import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

# Load the Iris dataset
iris = load_iris(as_frame=True)
X = iris.data[['sepal length (cm)', 'sepal width (cm)']]
y = (iris.target == 0)
# Binary classification (Setosa vs Not Setosa)  
per_clf = Perceptron(max_iter=1000, random_state=42)
per_clf.fit(X, y)

X_new =[[2,0.5], [3, 1]]
y_pred = per_clf.predict(X_new)
print("Predictions:", y_pred)
print("Predicted probabilities:", per_clf.decision_function(X_new))
print("Model coefficients:", per_clf.coef_)
print("Model intercept:", per_clf.intercept_)
print("Model score:", per_clf.score(X, y))
print("Model accuracy:", per_clf.score(X, y))
print("Model n_iter:", per_clf.n_iter)
print("Model n_iter_no_change:", per_clf.n_iter_no_change)