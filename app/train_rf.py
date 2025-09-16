# app/train_rf.py
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
    print("Fitting preprocessor...")
    pre = Preprocessor()
    pre.fit(df)
    print("Transforming data...")
    X = pre.transform(df)
    print("Features used for training:", list(X.columns))
    if "Traffic_Volume" not in df.columns and "Traffic Volume" in df.columns:
        df = df.rename(columns={"Traffic Volume": "Traffic_Volume"})
    if "Traffic_Volume" not in df.columns:
        raise ValueError("Target 'Traffic_Volume' not found in data.")

    y = df["Traffic_Volume"].values

    print("Split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    print("Training RandomForest...")
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("Evaluating...")
    preds = model.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, preds))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
    print("R2:", r2_score(y_test, preds))

    # save
    ensure_dir(MODEL_DIR)
    model_path = os.path.join(MODEL_DIR, "rf_traffic.pkl")
    preproc_path = os.path.join(MODEL_DIR, "preprocessor.joblib")
    joblib.dump(model, model_path)
    pre.save(preproc_path)
    print("Saved model ->", model_path)
    print("Saved preprocessor ->", preproc_path)

if __name__ == "__main__":
    train_and_save()
    print("Done.")