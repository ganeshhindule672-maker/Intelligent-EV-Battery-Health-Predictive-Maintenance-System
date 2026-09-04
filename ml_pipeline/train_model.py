"""
ml_pipeline/train_model.py

Trains Machine Learning models for EV Battery Life & Health Prediction:
  - Model 1: RUL (Remaining Useful Life in Cycles) Predictor
  - Model 2: SoH (State of Health %) Predictor

Evaluates using R2 score, MAE, RMSE, and extracts Feature Importances.
Saves artifacts to models/ directory.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

def train_battery_models(data_path="data/battery_aging_dataset.csv", output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Normalize column names to lowercase and strip whitespace
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Check required columns
    required_cols = ['voltage', 'current', 'temperature', 'capacity', 'cycle', 
                     'time', 'resistance', 'impedance', 'soc']
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in dataset: '{col}'")
            
    # If age or soh are missing, infer or calculate baseline
    if 'age' not in df.columns:
        df['age'] = df['cycle'] * 0.75
    if 'soh' not in df.columns:
        nominal_cap = df['capacity'].max()
        df['soh'] = (df['capacity'] / nominal_cap) * 100.0
    if 'rul' not in df.columns:
        # If RUL column is not present, calculate based on battery_id max cycle or global threshold
        if 'battery_id' in df.columns:
            max_cycles = df.groupby('battery_id')['cycle'].transform('max')
            df['rul'] = max_cycles - df['cycle']
        else:
            max_cycle = df['cycle'].max()
            df['rul'] = np.maximum(0, max_cycle - df['cycle'])

    # Feature lists
    soh_features = ['voltage', 'current', 'temperature', 'capacity', 'cycle', 
                    'time', 'resistance', 'impedance', 'soc', 'age']
    
    rul_features = ['voltage', 'current', 'temperature', 'capacity', 'cycle', 
                    'time', 'resistance', 'impedance', 'soc', 'soh', 'age']

    # -------------------------------------------------------------
    # 1. Train State of Health (SoH) Model
    # -------------------------------------------------------------
    print("\n--- Training State of Health (SoH) Predictor ---")
    X_soh = df[soh_features].fillna(df[soh_features].median())
    y_soh = df['soh']
    
    X_train_soh, X_test_soh, y_train_soh, y_test_soh = train_test_split(
        X_soh, y_soh, test_size=0.2, random_state=42
    )
    
    soh_model = GradientBoostingRegressor(
        n_estimators=180, 
        learning_rate=0.08, 
        max_depth=5, 
        random_state=42
    )
    soh_model.fit(X_train_soh, y_train_soh)
    
    y_pred_soh = soh_model.predict(X_test_soh)
    soh_r2 = float(r2_score(y_test_soh, y_pred_soh))
    soh_mae = float(mean_absolute_error(y_test_soh, y_pred_soh))
    soh_rmse = float(root_mean_squared_error(y_test_soh, y_pred_soh))
    
    print(f"SoH Model Evaluation -> R^2: {soh_r2:.4f} | MAE: {soh_mae:.4f}% | RMSE: {soh_rmse:.4f}%")

    # -------------------------------------------------------------
    # 2. Train Remaining Useful Life (RUL) Model
    # -------------------------------------------------------------
    print("\n--- Training Remaining Useful Life (RUL) Predictor ---")
    X_rul = df[rul_features].fillna(df[rul_features].median())
    y_rul = df['rul']
    
    X_train_rul, X_test_rul, y_train_rul, y_test_rul = train_test_split(
        X_rul, y_rul, test_size=0.2, random_state=42
    )
    
    rul_model = RandomForestRegressor(
        n_estimators=180, 
        max_depth=12, 
        n_jobs=-1, 
        random_state=42
    )
    rul_model.fit(X_train_rul, y_train_rul)
    
    y_pred_rul = rul_model.predict(X_test_rul)
    rul_r2 = float(r2_score(y_test_rul, y_pred_rul))
    rul_mae = float(mean_absolute_error(y_test_rul, y_pred_rul))
    rul_rmse = float(root_mean_squared_error(y_test_rul, y_pred_rul))
    
    print(f"RUL Model Evaluation -> R^2: {rul_r2:.4f} | MAE: {rul_mae:.2f} cycles | RMSE: {rul_rmse:.2f} cycles")

    # -------------------------------------------------------------
    # Feature Importances
    # -------------------------------------------------------------
    rul_importance = {
        feat: round(float(imp), 4) 
        for feat, imp in zip(rul_features, rul_model.feature_importances_)
    }
    # Sort descending
    rul_importance = dict(sorted(rul_importance.items(), key=lambda item: item[1], reverse=True))

    soh_importance = {
        feat: round(float(imp), 4) 
        for feat, imp in zip(soh_features, soh_model.feature_importances_)
    }
    soh_importance = dict(sorted(soh_importance.items(), key=lambda item: item[1], reverse=True))

    # -------------------------------------------------------------
    # Save Artifacts
    # -------------------------------------------------------------
    joblib.dump(soh_model, os.path.join(output_dir, "battery_soh_model.joblib"))
    joblib.dump(rul_model, os.path.join(output_dir, "battery_rul_model.joblib"))
    
    metadata = {
        "soh_features": soh_features,
        "rul_features": rul_features,
        "metrics": {
            "soh": {
                "r2": round(soh_r2, 4),
                "mae": round(soh_mae, 4),
                "rmse": round(soh_rmse, 4)
            },
            "rul": {
                "r2": round(rul_r2, 4),
                "mae": round(rul_mae, 2),
                "rmse": round(rul_rmse, 2)
            }
        },
        "feature_importances": {
            "rul": rul_importance,
            "soh": soh_importance
        },
        "dataset_info": {
            "rows": len(df),
            "columns": list(df.columns)
        }
    }
    
    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\n-> Successfully saved models and metadata to '{output_dir}'")
    return metadata

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_file = os.path.join(project_root, "data", "battery_aging_dataset.csv")
    models_dir = os.path.join(project_root, "models")
    train_battery_models(data_file, models_dir)
