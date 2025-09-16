# app/api.py
import os
import joblib
from flask import Flask, request, jsonify
import pandas as pd
from app.preprocessing import Preprocessor


# Helper: traffic_count → congestion level
def get_congestion_level(traffic_count):
    if traffic_count <= 2000:
        return "Low"
    elif traffic_count <= 5000:
        return "Medium"
    elif traffic_count <= 8000:
        return "High"
    else:
        return "Severe"


app = Flask(__name__)

# --------------------------------------------------------------------
# Load models and preprocessor once at startup
# --------------------------------------------------------------------
# --- Define paths relative to this file's location ---
# This makes the app runnable from any directory
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_ROOT, ".."))

MODEL_TRAFFIC_PATH = os.path.join(PROJECT_ROOT, "models", "rf_traffic.pkl")
MODEL_SPEED_PATH = os.path.join(PROJECT_ROOT, "models", "rf_speed.pkl")
PREPROC_PATH = os.path.join(PROJECT_ROOT, "models", "preprocessor.joblib")

# Load models + preprocessor
try:
    model_traffic = joblib.load(MODEL_TRAFFIC_PATH)
    model_speed = joblib.load(MODEL_SPEED_PATH)
    preprocessor = Preprocessor.load(PREPROC_PATH)
except FileNotFoundError as e:
    print(f"Error loading models: {e}")
    print("Please ensure you have trained the models by running 'train_rf.py' and 'train_rf_speed.py' first.")
    exit()


# --------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Input JSON (user should NOT send congestion_level or avg_speed):
    {
      "Date": "2025-09-09 08:00:00",
      "Area_Name": "Indiranagar",
      "Road_Intersection_Name": "100ft road",
      "Weather_Conditions": "Clear",
      "Roadwork_and_Construction_Activity": "No"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    # --- Remove forbidden keys if present ---
    for forbidden in ["Average_Speed", "Congestion_Level"]:
        if forbidden in data:
            data.pop(forbidden)

    # Build single-row DataFrame
    df = pd.DataFrame([data])
    X = preprocessor.transform(df)

    # Predict traffic count
    traffic_pred = model_traffic.predict(X)
    traffic_count = int(traffic_pred[0])

    # Predict average speed
    speed_pred = model_speed.predict(X)
    avg_speed = float(speed_pred[0])

    # Derive congestion level from traffic count
    congestion = get_congestion_level(traffic_count)

    return jsonify({
        "traffic_count": traffic_count,
        "average_speed": avg_speed,
        "congestion_level": congestion
    })


if __name__ == "__main__":
    # Run dev server
    app.run(host="0.0.0.0", port=5000, debug=True)
