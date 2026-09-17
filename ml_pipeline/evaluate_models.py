"""
ml_pipeline/evaluate_models.py

Comprehensive model evaluation and accuracy benchmarking script
for the Intelligent EV Battery Health & Predictive Maintenance System.
Evaluates both SOH and RUL models using:
  - R² Score (Coefficient of Determination)
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Percentage Error (MAPE)
  - Overall Accuracy (%)
  - 5-Fold Cross Validation Scores
  - Health State Classification Accuracy (4-Tier Health Status)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    accuracy_score,
    classification_report
)

def evaluate_and_train(
    data_path="data/nasa_real_battery_dataset.csv",
    output_dir="models"
):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    print(f"[INFO] Loading authentic dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"[INFO] Dataset shape: {df.shape} ({len(df)} cycles, {df['battery_id'].nunique()} batteries)")

    # Features
    soh_features = ['voltage', 'current', 'temperature', 'capacity', 'cycle', 'time', 'resistance', 'impedance', 'soc', 'age']
    rul_features = ['voltage', 'current', 'temperature', 'capacity', 'cycle', 'time', 'resistance', 'impedance', 'soc', 'soh', 'age']

    X_soh = df[soh_features]
    y_soh = df['soh']

    X_rul = df[rul_features]
    y_rul = df['rul']

    # Train / Test split (80/20 with fixed random state for reproducibility)
    X_train_soh, X_test_soh, y_train_soh, y_test_soh = train_test_split(
        X_soh, y_soh, test_size=0.20, random_state=42
    )

    X_train_rul, X_test_rul, y_train_rul, y_test_rul = train_test_split(
        X_rul, y_rul, test_size=0.20, random_state=42
    )

    # -------------------------------------------------------------
    # 1. Evaluate State of Health (SOH) Model
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("[1/3] EVALUATING STATE OF HEALTH (SOH %) MODEL")
    print("="*60)

    soh_model = GradientBoostingRegressor(
        n_estimators=220,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.9,
        random_state=42
    )
    soh_model.fit(X_train_soh, y_train_soh)

    # Predictions
    y_pred_soh_train = soh_model.predict(X_train_soh)
    y_pred_soh_test = soh_model.predict(X_test_soh)

    soh_r2_train = float(r2_score(y_train_soh, y_pred_soh_train))
    soh_r2_test = float(r2_score(y_test_soh, y_pred_soh_test))
    soh_mae_test = float(mean_absolute_error(y_test_soh, y_pred_soh_test))
    soh_rmse_test = float(root_mean_squared_error(y_test_soh, y_pred_soh_test))
    soh_mape_test = float(mean_absolute_percentage_error(y_test_soh, y_pred_soh_test))
    soh_accuracy_pct = float(max(0.0, (1.0 - soh_mape_test) * 100.0))

    # 5-Fold Cross Validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    soh_cv_scores = cross_val_score(soh_model, X_soh, y_soh, cv=kfold, scoring='r2')

    print(f"* SOH R^2 Score (Test)      : {soh_r2_test * 100:.2f}% ({soh_r2_test:.4f})")
    print(f"* SOH Prediction Accuracy   : {soh_accuracy_pct:.2f}% (based on 100 - MAPE)")
    print(f"* SOH Mean Absolute Error   : {soh_mae_test:.4f}%")
    print(f"* SOH Root Mean Sq Error    : {soh_rmse_test:.4f}%")
    print(f"* SOH 5-Fold CV Mean R^2    : {np.mean(soh_cv_scores):.4f} (+/- {np.std(soh_cv_scores):.4f})")

    # -------------------------------------------------------------
    # 2. Evaluate Remaining Useful Life (RUL) Model
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("[2/3] EVALUATING REMAINING USEFUL LIFE (RUL) MODEL")
    print("="*60)

    rul_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=14,
        min_samples_split=3,
        n_jobs=-1,
        random_state=42
    )
    rul_model.fit(X_train_rul, y_train_rul)

    # Predictions
    y_pred_rul_train = rul_model.predict(X_train_rul)
    y_pred_rul_test = rul_model.predict(X_test_rul)

    rul_r2_train = float(r2_score(y_train_rul, y_pred_rul_train))
    rul_r2_test = float(r2_score(y_test_rul, y_pred_rul_test))
    rul_mae_test = float(mean_absolute_error(y_test_rul, y_pred_rul_test))
    rul_rmse_test = float(root_mean_squared_error(y_test_rul, y_pred_rul_test))
    
    # MAPE for non-zero RUL
    nonzero_idx = y_test_rul > 0
    rul_mape_test = float(mean_absolute_percentage_error(y_test_rul[nonzero_idx], y_pred_rul_test[nonzero_idx]))
    rul_accuracy_pct = float(max(0.0, (1.0 - rul_mape_test) * 100.0))

    # 5-Fold Cross Validation
    rul_cv_scores = cross_val_score(rul_model, X_rul, y_rul, cv=kfold, scoring='r2')

    print(f"* RUL R^2 Score (Test)      : {rul_r2_test * 100:.2f}% ({rul_r2_test:.4f})")
    print(f"* RUL Prediction Accuracy   : {rul_accuracy_pct:.2f}% (based on 100 - MAPE)")
    print(f"* RUL Mean Absolute Error   : {rul_mae_test:.2f} cycles")
    print(f"* RUL Root Mean Sq Error    : {rul_rmse_test:.2f} cycles")
    print(f"* RUL 5-Fold CV Mean R^2    : {np.mean(rul_cv_scores):.4f} (+/- {np.std(rul_cv_scores):.4f})")

    # -------------------------------------------------------------
    # 3. Health Status Classification Accuracy
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("[3/3] BATTERY HEALTH STATUS CLASSIFICATION ACCURACY")
    print("="*60)

    def to_health_tier(soh_val):
        if soh_val >= 90.0:
            return "EXCELLENT"
        elif soh_val >= 80.0:
            return "GOOD"
        elif soh_val >= 70.0:
            return "DEGRADED"
        else:
            return "CRITICAL"

    y_test_tier_actual = [to_health_tier(val) for val in y_test_soh]
    y_test_tier_pred = [to_health_tier(val) for val in y_pred_soh_test]
    classification_acc = float(accuracy_score(y_test_tier_actual, y_test_tier_pred)) * 100.0

    print(f"* 4-Tier Health Status Accuracy: {classification_acc:.2f}%\n")
    report_dict = classification_report(y_test_tier_actual, y_test_tier_pred, output_dict=True)
    print(classification_report(y_test_tier_actual, y_test_tier_pred))

    # -------------------------------------------------------------
    # 4. Feature Importances
    # -------------------------------------------------------------
    rul_importance = {
        feat: round(float(imp), 4)
        for feat, imp in zip(rul_features, rul_model.feature_importances_)
    }
    rul_importance = dict(sorted(rul_importance.items(), key=lambda item: item[1], reverse=True))

    soh_importance = {
        feat: round(float(imp), 4)
        for feat, imp in zip(soh_features, soh_model.feature_importances_)
    }
    soh_importance = dict(sorted(soh_importance.items(), key=lambda item: item[1], reverse=True))

    # -------------------------------------------------------------
    # 5. Save Artifacts & Accuracy Scorecard
    # -------------------------------------------------------------
    joblib.dump(soh_model, os.path.join(output_dir, "battery_soh_model.joblib"))
    joblib.dump(rul_model, os.path.join(output_dir, "battery_rul_model.joblib"))

    metadata = {
        "soh_features": soh_features,
        "rul_features": rul_features,
        "metrics": {
            "soh": {
                "r2": round(soh_r2_test, 4),
                "mae": round(soh_mae_test, 4),
                "rmse": round(soh_rmse_test, 4),
                "mape": round(soh_mape_test * 100, 2),
                "accuracy_score": round(soh_accuracy_pct, 2),
                "cv_5fold_mean_r2": round(float(np.mean(soh_cv_scores)), 4)
            },
            "rul": {
                "r2": round(rul_r2_test, 4),
                "mae": round(rul_mae_test, 2),
                "rmse": round(rul_rmse_test, 2),
                "mape": round(rul_mape_test * 100, 2),
                "accuracy_score": round(rul_accuracy_pct, 2),
                "cv_5fold_mean_r2": round(float(np.mean(rul_cv_scores)), 4)
            },
            "health_classification": {
                "accuracy_percentage": round(classification_acc, 2),
                "tiers": ["EXCELLENT", "GOOD", "DEGRADED", "CRITICAL"]
            }
        },
        "feature_importances": {
            "rul": rul_importance,
            "soh": soh_importance
        },
        "dataset_info": {
            "name": "NASA Ames Prognostics Center of Excellence (PCoE) Real Battery Aging Dataset",
            "rows": len(df),
            "batteries_count": int(df['battery_id'].nunique()),
            "columns": list(df.columns)
        }
    }

    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    # Save detailed accuracy report
    acc_report_path = os.path.join(output_dir, "accuracy_report.json")
    with open(acc_report_path, "w") as f:
        json.dump({
            "model_metadata": metadata,
            "classification_report": report_dict
        }, f, indent=4)

    print(f"-> Saved evaluated models and accuracy metadata to '{output_dir}'")
    return metadata

if __name__ == "__main__":
    evaluate_and_train()
