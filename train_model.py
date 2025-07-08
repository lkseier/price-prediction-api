import joblib
import json
import os

#Get the absolute path to the model directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

#Load the model
model_path = os.path.join((__file__), "model", "best_model.pkl")
model = joblib.load("model/best_model.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

#Load the features
features_path = os.path.join(MODEL_DIR, "features.json")
with open(features_path, "r") as f:
    feature_names = json.load(f)

