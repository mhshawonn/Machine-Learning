import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

def plot_sets(train_X, train_y, val_X, val_y):
    # Extract raw x values without scaling
    train_x = train_X[:, 1]  
    val_x = val_X[:, 1]  

    # Plot the training set
    plt.scatter(train_x, train_y, color='blue', label='Train')
    plt.title("Training Set")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig("plots/training.png")
    plt.clf()

    # Plot the validation set
    plt.scatter(val_x, val_y, color='orange', label='Validation')
    plt.title("Validation Set")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig("plots/validation.png")
    plt.clf()

def plot_error_curve(train_errors, val_errors):
    plt.plot(train_errors, label="Train Error")
    plt.plot(val_errors, label="Validation Error")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Error Curve")
    plt.legend()
    plt.savefig("plots/error_curve.png")
    plt.clf()

def plot_final_fit(df, theta):
    # Directly use raw x values without scaling
    x_values = df["x"].values
    X_plot = np.c_[np.ones(x_values.shape), x_values]  # Add a column of ones for the bias term
    y_pred = X_plot @ theta

    # Plot the dataset and the regression line
    plt.scatter(df["x"], df["y"], label="Data")
    plt.plot(df["x"], y_pred, color="red", label="Regression Line")
    plt.title("Final Regression Fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig("plots/final_fit.png")
    plt.clf()
