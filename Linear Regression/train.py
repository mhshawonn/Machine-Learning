from model import gradient_descent, compute_cost

def train(train_X, train_y, val_X, val_y, alpha=0.01, epochs=200):
    return gradient_descent(train_X, train_y, val_X, val_y, alpha, epochs)

def evaluate(theta, train_errors, val_errors):
    print(f"Final Training Error: {train_errors[-1]:.4f}")
    print(f"Final Validation Error: {val_errors[-1]:.4f}")

    best_val = min(val_errors)
    best_epoch = val_errors.index(best_val)
    print(f"Best Validation Error: {best_val:.4f}")
    print(f"Corresponding Training Error: {train_errors[best_epoch]:.4f}")
