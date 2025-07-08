from fastapi import FastAPI
from flask import Flask, request, jsonify
import streamlit as st
import joblib, os, json


model_path = os.path.join("models", "best_model.pkl")
model = joblib.load(model_path)

with open(os.path.join("models", "features.json"), "r") as f:
    feature_names = json.load(f)

app = Flask(__name__)

# Route: GET / => Check if server is alive
@app.route('/', methods=['GET'])
def home():
    return "alive", 200

# Route: GET /predict => Describe expected POST format
@app.route('/predict', methods=['GET'])
def predict_info():
    return jsonify({"message": 'Send a POST request with JSON in the following format: {"feature1": value, "feature2": value, ...}'}), 200

# Route: POST /predict => Receive house data and return a dummy prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON received"}), 400

    # Example: Replace with actual model prediction logic
    # Ensure data keys match feature_names
    input_data = [data[feature] for feature in feature_names]
    prediction = model.predict([input_data])[0]
    return jsonify({"prediction": prediction}), 200

if __name__ == '__main__':
    app.run(debug=True)