from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Simulate a prediction using NumPy
    prediction = np.dot([0.5, 1.2, 0.8], list(data.values()))  # Dummy prediction
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
