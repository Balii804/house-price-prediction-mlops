from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Home Page"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Ensure the correct number of features
    expected_features = [0.5, 1.2, 0.8]
    features = list(data.values())
    if len(features) != len(expected_features):
        return jsonify({'error': 'Invalid input shape'}), 400
    prediction = np.dot(expected_features, features)  # Dummy prediction
    return jsonify({'prediction': prediction})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Simulate a prediction using NumPy
    prediction = np.dot([0.5, 1.2, 0.8], list(data.values()))  # Dummy prediction
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
# Load the trained model weights and normalization parameters
theta = np.load('model_weights.npy')
X_mean = np.load('model_mean.npy')
X_std = np.load('model_std.npy')

# Prediction function
def predict_price(area, basement, garage):
    X = np.array([area, basement, garage])
    X_normalized = (X - X_mean) / X_std
    X_normalized = np.hstack(([1], X_normalized))
    price = X_normalized.dot(theta)
    return price

# Define a prediction API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    area = data['area']
    basement = data['basement']
    garage = data['garage']
    
    price = predict_price(area, basement, garage)
    return jsonify({'predicted_price': price})

if __name__ == '__main__':
    app.run(debug=True)


