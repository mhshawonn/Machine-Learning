from data_utils import generate_data, process_data
from train import train, evaluate
from visualize import plot_sets, plot_error_curve, plot_final_fit

def main():
    # Generate data and save to CSV
    generate_data()
    
    # Process the data (split into train/test and prepare features)
    train_X, train_y, val_X, val_y, df = process_data()
    
    # Plot the training and validation sets
    plot_sets(train_X, train_y, val_X, val_y)
    
    # Train the model
    theta, train_errors, val_errors = train(train_X, train_y, val_X, val_y)
    
    # Evaluate the model
    evaluate(theta, train_errors, val_errors)
    
    # Plot error curves (Training and Validation Errors)
    plot_error_curve(train_errors, val_errors)
    
    # Plot the final regression line
    plot_final_fit(df, theta)
    
    # Print the learned parameters (theta)
    print("Learned parameters (theta):")
    print(theta)

if __name__ == "__main__":
    main()
