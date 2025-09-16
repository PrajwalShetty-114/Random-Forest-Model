# app/train_rf_speed.py
import os
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from app.preprocessing import Preprocessor

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bangalore_traffic.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
ensure_dir = lambda p: os.makedirs(p, exist_ok=True)

def train_and_save():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # --- Load the preprocessor saved by train_rf.py ---
    # This ensures both models use the exact same features and encodings.
    preproc_path = os.path.join(MODEL_DIR, "preprocessor.joblib")
    print(f"Loading preprocessor from {preproc_path}...")
    pre = Preprocessor.load(preproc_path)
    print("Transforming data...")
    X = pre.transform(df)

    # Check target column
    if "Average_Speed" not in df.columns:
        raise ValueError("Target 'Average_Speed' not found in data.")

    y = df["Average_Speed"].values

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    print("Training RandomForest for Average Speed...")
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    preds = model.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, preds))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
    print("R2:", r2_score(y_test, preds))

    # Save model and preprocessor
    ensure_dir(MODEL_DIR)
    model_path = os.path.join(MODEL_DIR, "rf_speed.pkl") 
    joblib.dump(model, model_path)
    
    print("Saved Average Speed model ->", model_path)

if __name__ == "__main__":
    train_and_save()
    print("Done.")
