import numpy as np

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

    predictions = predict(model, X_new)

    
    print(f"Predictions: {predictions}")
