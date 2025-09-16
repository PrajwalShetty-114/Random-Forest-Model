# app/preprocessing.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
import joblib

# Paths (relative usage)
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

class Preprocessor:
    def __init__(self):
        # one-hot weather (many categories)
        self.weather_ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        # ordinal encoders for Area and Road (stable integer encoding)
        self.area_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.road_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        
        # fitted column order (after transform)
        self.feature_columns = None

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        # Normalize common names (adjust to match your CSV)
        rename_map = {
            "Area Name": "Area_Name",
            "Road/Intersection Name": "Road_Intersection_Name",
            "Traffic Volume": "Traffic_Volume",
            "Weather Conditions": "Weather_Conditions",
            "Roadwork and Construction Activity": "Roadwork_and_Construction_Activity",
            # add other renames if your CSV uses different names
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        return df

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        df = self._normalize_column_names(df)

        # Make sure columns exist (create default if missing)
        if "Roadwork_and_Construction_Activity" not in df.columns:
            df["Roadwork_and_Construction_Activity"] = "No"

        # Datetime features
        if "Date" in df.columns:
            # try to parse; accept Excel serial or ISO. Here we try flexible parse
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        else:
            df["Date"] = pd.NaT

        df["hour_of_day"] = df["Date"].dt.hour.fillna(0).astype(int)
        df["day_of_week"] = df["Date"].dt.dayofweek.fillna(0).astype(int)
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # Fit encoders on columns that exist
        if "Weather_Conditions" in df.columns:
            self.weather_ohe.fit(df[["Weather_Conditions"]].fillna("Unknown"))
        if "Area_Name" in df.columns:
            self.area_enc.fit(df[["Area_Name"]].fillna("Unknown"))
        if "Road_Intersection_Name" in df.columns:
            self.road_enc.fit(df[["Road_Intersection_Name"]].fillna("Unknown"))

        # build feature_columns by running transform on a small sample
        # This ensures the feature order is saved correctly
        sample = df.head(1)
        transformed = self.transform(sample)
        self.feature_columns = transformed.columns.tolist()
        return self

    def transform(self, df: pd.DataFrame, fit_mode=False) -> pd.DataFrame:
        # fit_mode True only used inside fit() above to avoid double-fitting
        df = df.copy()
        df = self._normalize_column_names(df)

        # ensure fallback columns
        if "Roadwork_and_Construction_Activity" not in df.columns:
            df["Roadwork_and_Construction_Activity"] = "No"

        # parse date safely
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        else:
            df["Date"] = pd.NaT

        df["hour_of_day"] = df["Date"].dt.hour.fillna(0).astype(int)
        df["day_of_week"] = df["Date"].dt.dayofweek.fillna(0).astype(int)
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # binary encode roadwork (Yes/No)
        df["Roadwork_flag"] = df["Roadwork_and_Construction_Activity"].fillna("No").astype(str).str.lower().apply(lambda x: 1 if "yes" in x else 0)

        # Weather one-hot
        weather_cols = []
        if "Weather_Conditions" in df.columns:
            weather_arr = self.weather_ohe.transform(df[["Weather_Conditions"]].fillna("Unknown"))
            weather_cols = [f"Weather_{c}" for c in self.weather_ohe.categories_[0]]
            weather_df = pd.DataFrame(weather_arr, columns=weather_cols, index=df.index)
            df = pd.concat([df, weather_df], axis=1)

        # --- Area / Road ordinal ---
        if "Area_Name" in df.columns:
            df["Area_encoded"] = self.area_enc.transform(df[["Area_Name"]].fillna("Unknown"))
        else:
            df["Area_encoded"] = -1
        
        if "Road_Intersection_Name" in df.columns:
            df["Road_encoded"] = self.road_enc.transform(df[["Road_Intersection_Name"]].fillna("Unknown"))
        else:
            df["Road_encoded"] = -1

        # --- Define explicit feature set ---
        # These are the only columns the models will be trained on and expect for prediction.
        # We explicitly EXCLUDE 'Average_Speed' and 'Congestion_Level' as they are targets or derived.
        final_cols = [
            "hour_of_day", "day_of_week", "is_weekend", "Roadwork_flag", 
            "Area_encoded", "Road_encoded"
        ]
        # Add other numeric features from the dataset if they are available at prediction time
        for col in ["Travel_Time_Index", "Road_Capacity_Utilization", "Incident_Reports", "Environmental_Impact", "Public_Transport_Usage", "Traffic_Signal_Compliance", "Parking_Usage", "Pedestrian_and_Cyclist_Count"]:
            if col in df.columns:
                final_cols.append(col)

        # add weather columns
        final_cols += [c for c in weather_cols if c not in final_cols]

        # ensure each exists (fill missing with 0)
        out = pd.DataFrame(index=df.index)
        for c in final_cols:
            out[c] = df[c] if c in df.columns else 0

        # Reorder columns to match the order during fit
        if self.feature_columns:
            # For prediction, ensure all columns are present and in the correct order
            for col in self.feature_columns:
                if col not in out.columns:
                    out[col] = 0 # Add missing columns with a default value
            out = out[self.feature_columns]

        return out

    def save(self, filepath: str):
        ensure_dir(os.path.dirname(filepath))
        joblib.dump({
            "weather_ohe": self.weather_ohe,
            "area_enc": self.area_enc,
            "road_enc": self.road_enc,
            "feature_columns": self.feature_columns
        }, filepath)

    @classmethod
    def load(cls, filepath: str):
        payload = joblib.load(filepath)
        obj = cls()
        obj.weather_ohe = payload["weather_ohe"]
        obj.feature_columns = payload.get("feature_columns", None)

        # --- Backward compatibility for loading encoders ---
        # Check for new keys first (which load the entire encoder object)
        if "area_enc" in payload and "road_enc" in payload:
            print("INFO: Loading preprocessor in new, recommended format.")
            obj.area_enc = payload["area_enc"]
            obj.road_enc = payload["road_enc"]
        # Fallback to old keys if new ones are not found
        else:
            print("INFO: Loading preprocessor with backward compatibility for older format.")
            # Re-create ordinal encoders from saved categories
            obj.area_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            area_categories = payload.get("area_categories")
            if area_categories is not None:
                obj.area_enc.categories_ = area_categories

            obj.road_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            road_categories = payload.get("road_categories")
            if road_categories is not None:
                obj.road_enc.categories_ = road_categories

        return obj
