import numpy as np

def compute_cost(X, y, theta):
    m = len(y)
    error = X @ theta - y
    return (1 / m) * np.sum(error ** 2)

def gradient_descent(X, y, X_val, y_val, alpha=0.0005, epochs=500):
    m, n = X.shape
    theta = np.zeros((n, 1))
    train_errors = []
    val_errors = []

    for _ in range(epochs):
        # Compute the error and gradient
        error = X @ theta - y
        grad = (2 / m) * X.T @ error
        theta -= alpha * grad

        # Append the cost for training and validation
        train_errors.append(compute_cost(X, y, theta))
        val_errors.append(compute_cost(X_val, y_val, theta))

    return theta, train_errors, val_errors
