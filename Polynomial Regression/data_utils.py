import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement

# 1. Load Data (Here synthetic)
def load_data():
    np.random.seed(42)
    X = 6 * np.random.rand(100, 1) - 3
    y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)
    return X, y

# 2. Process Data (Add Polynomial Features + Bias Term)
def process_data(X, degree):
    m, n = X.shape
    features = [np.ones((m, 1))]  # Start with bias term (x0=1)

    for d in range(1, degree + 1):
        for index in combinations_with_replacement(range(n), d):
            new_feature = np.prod(X[:, index], axis=1).reshape(-1, 1)
            features.append(new_feature)

    return np.hstack(features)

# 3. Compute Cost (MSE)
def compute_cost(X, y, theta):
    m = len(y)
    error = X @ theta - y
    return (1/(2*m)) * np.sum(error**2)

# 4. Gradient Descent
def gradient_descent(X, y, alpha=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros((n, 1))
    cost_history = []

    for _ in range(epochs):
        gradients = (1/m) * X.T @ (X @ theta - y)
        theta -= alpha * gradients
        cost_history.append(compute_cost(X, y, theta))
    
    return theta, cost_history

# 5. Train Model
def train(X, y, degree, alpha=0.1, epochs=1000):
    X_poly = process_data(X, degree)
    theta, cost_history = gradient_descent(X_poly, y, alpha, epochs)
    return theta, cost_history

# 6. Predict
def predict(X, theta, degree):
    X_poly = process_data(X, degree)
    return X_poly @ theta

# 7. Evaluate Model
def evaluate(X, y, theta, degree):
    X_poly = process_data(X, degree)
    return compute_cost(X_poly, y, theta)

# 8. Plotting Utilities
def plot_predictions(X, y, theta, degree):
    plt.scatter(X, y, color="blue", label="Data Points")
    X_new = np.linspace(X.min()-1, X.max()+1, 100).reshape(100, 1)
    y_pred = predict(X_new, theta, degree)
    plt.plot(X_new, y_pred, color="red", label=f"Polynomial Regression (Degree {degree})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Polynomial Regression Fit")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_cost(cost_history):
    plt.plot(cost_history)
    plt.xlabel("Epoch")
    plt.ylabel("Cost (MSE)")
    plt.title("Cost Reduction Over Time")
    plt.grid(True)
    plt.show()

# 9. Main Program
def main():
    degree = 3
    X, y = load_data()
    
    theta, cost_history = train(X, y, degree, alpha=0.1, epochs=1000)
    
    final_cost = evaluate(X, y, theta, degree)
    
    print("Final Training Error (MSE):", final_cost)
    print("Learned Parameters (theta):\n", theta)
    
    plot_predictions(X, y, theta, degree)
    plot_cost(cost_history)

if __name__ == "__main__":
    main()
