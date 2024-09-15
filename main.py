import pandas as pd
import numpy as np

# Update file paths if necessary
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Select features and target from training data
X_train_raw = train_data[['GrLivArea', 'TotalBsmtSF', 'GarageArea']].values
y_train = train_data['SalePrice'].values

# Preprocess the data
def preprocess_data(X):
    # Normalize the data
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_normalized = (X - X_mean) / X_std
    X_normalized = np.hstack((np.ones((X_normalized.shape[0], 1)), X_normalized))  # Add bias term
    return X_normalized, X_mean, X_std

# Preprocess training data
X_train, X_mean, X_std = preprocess_data(X_train_raw)

# Preprocess test data
X_test_raw = test_data[['GrLivArea', 'TotalBsmtSF', 'GarageArea']].values
X_test, _, _ = preprocess_data(X_test_raw)

# Split training data into training and validation sets
train_size = int(0.8 * len(X_train))
X_train_split, X_val_split = X_train[:train_size], X_train[train_size:]
y_train_split, y_val_split = y_train[:train_size], y_train[train_size:]

# Linear regression model using gradient descent
def train_linear_regression(X, y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)
    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradients = 2/m * X.T.dot(errors)
        theta -= learning_rate * gradients
    return theta

# Train the model
theta = train_linear_regression(X_train_split, y_train_split)

# Prediction function
def predict(X, theta):
    return X.dot(theta)

# Save the trained model (weights) and preprocessing parameters for use in Flask
np.save('model_weights.npy', theta)
np.save('model_mean.npy', X_mean)
np.save('model_std.npy', X_std)

# Optionally, you can test the model on the validation set and print the result
val_predictions = predict(X_val_split, theta)
val_error = np.mean((val_predictions - y_val_split) ** 2)
print(f'Validation Mean Squared Error: {val_error}')

# Create a synthetic dataset
def create_dataset():
    # Features: [Size (sq ft), Number of Bedrooms, Age (years)]
    data = np.array([
        [1500, 3, 10],
        [1800, 4, 15],
        [2000, 3, 8],
        [2500, 5, 20],
        [3000, 4, 12]
    ])
    
    # Target: Prices in thousands
    target = np.array([400, 500, 450, 600, 650])
    
    return data, target

# Dummy dataset preprocessing
def preprocess(data):
    # Add a bias column with ones
    ones = np.ones((data.shape[0], 1))
    return np.hstack([ones, data])

# Dummy model training using linear regression
def train_model(X, y):
    # Use the Normal Equation: weights = (X^T * X)^-1 * X^T * y
    X_transpose = X.T
    weights = np.linalg.pinv(X_transpose @ X) @ X_transpose @ y
    return weights

# Make predictions using the trained model
def predict(model, X):
    return X @ model

# Save the model weights to a file
def save_model_weights(model, filename):
    np.save(filename, model)

# Load the model weights from a file
def load_model_weights(filename):
    return np.load(filename)

if _name_ == "_main_":
    # Create the dataset
    data, target = create_dataset()
    
    # Preprocess the data
    X = preprocess(data)
    
    # Train the model
    model = train_model(X, target)
    print(f"Model weights: {model}")
    
    # Save the model weights to a file
    save_model_weights(model, 'model_weights.npy')
    
    # Load the model weights (if needed later)
    model_loaded = load_model_weights('model_weights.npy')
    print(f"Loaded model weights: {model_loaded}")
    
    # Predict on new data
    new_data = np.array([
        [1600, 3, 12],
        [2200, 4, 10]
    ])
    X_new = preprocess(new_data)
    predictions = predict(model_loaded, X_new)
    
    print(f"Predictions: {predictions}")
