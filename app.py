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
if __name__ == "__main__":
    app.run(debug=True)
