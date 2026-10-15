"""
app.py

Flask Web Application for EV Battery Life & Health Prediction System.
Connects the ML prediction engine, batch dataset ingestion (NASA & CALCE),
and interactive visual analytics dashboard.
"""

import os
import json
import traceback
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename

from ml_pipeline.model_engine import BatteryHealthEngine
from ml_pipeline.train_model import train_battery_models

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ev-battery-intelligence-secret-key-2026'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64 MB max upload

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize ML Engine
engine = BatteryHealthEngine(models_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))

# Preset configurations for 1-click testing
SAMPLE_PRESETS = {
    "brand_new": {
        "name": "Brand New Cell (0 Cycles)",
        "voltage": 4.12,
        "current": 1.2,
        "temperature": 24.5,
        "capacity": 2.01,
        "cycle": 5,
        "time": 18000,
        "resistance": 48.0,
        "impedance": 0.038,
        "soc": 92.0,
        "soh": 99.5,
        "age": 4.0
    },
    "moderate_aging": {
        "name": "Mid-Life Cell (150 Cycles)",
        "voltage": 3.68,
        "current": 1.8,
        "temperature": 31.0,
        "capacity": 1.74,
        "cycle": 150,
        "time": 540000,
        "resistance": 76.5,
        "impedance": 0.058,
        "soc": 75.0,
        "soh": 86.5,
        "age": 115.0
    },
    "second_life": {
        "name": "Degraded Cell / Second Life (280 Cycles)",
        "voltage": 3.42,
        "current": 2.1,
        "temperature": 38.5,
        "capacity": 1.48,
        "cycle": 280,
        "time": 1008000,
        "resistance": 115.0,
        "impedance": 0.082,
        "soc": 60.0,
        "soh": 73.8,
        "age": 210.0
    },
    "critical_thermal": {
        "name": "Critical / Thermal Stress (360 Cycles)",
        "voltage": 3.15,
        "current": 2.4,
        "temperature": 48.2,
        "capacity": 1.28,
        "cycle": 360,
        "time": 1296000,
        "resistance": 158.0,
        "impedance": 0.115,
        "soc": 45.0,
        "soh": 63.5,
        "age": 275.0
    }
}

@app.route('/')
def index():
    metadata = engine.metadata
    return render_template('index.html', metadata=metadata, presets=SAMPLE_PRESETS)

@app.route('/batch')
def batch():
    # List available sample datasets
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    samples = []
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.endswith('.csv'):
                samples.append(f)
    return render_template('batch.html', samples=samples)

@app.route('/analytics')
def analytics():
    metadata = engine.metadata
    return render_template('analytics.html', metadata=metadata)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json(force=True)
        result = engine.predict_single(data)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 400

@app.route('/api/presets', methods=['GET'])
def api_presets():
    return jsonify({"success": True, "presets": SAMPLE_PRESETS})

@app.route('/api/upload-dataset', methods=['POST'])
def api_upload_dataset():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded."}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No selected file."}), 400
            
        if not file.filename.endswith('.csv'):
            return jsonify({"success": False, "error": "Only CSV files are supported."}), 400

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        # Inspect dataset
        df = pd.read_csv(save_path)
        df_cols_clean = [c.strip().lower() for c in df.columns]
        
        target_cols = ['voltage', 'current', 'temperature', 'capacity', 'cycle', 
                       'time', 'resistance', 'impedance', 'soc', 'soh']
        
        matched_cols = [col for col in target_cols if col in df_cols_clean]
        missing_cols = [col for col in target_cols if col not in df_cols_clean]

        # Preview head
        preview_data = df.head(10).to_dict(orient='records')
        
        # Summary statistics for numeric columns
        stats = df.describe().round(3).to_dict()

        return jsonify({
            "success": True,
            "filename": filename,
            "filepath": save_path,
            "rows": len(df),
            "columns": list(df.columns),
            "matched_cols": matched_cols,
            "missing_cols": missing_cols,
            "preview": preview_data,
            "stats": stats
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/retrain', methods=['POST'])
def api_retrain():
    try:
        data = request.get_json(silent=True) or {}
        filename = data.get('filename')
        
        if filename:
            target_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(target_path):
                # Check in data dir
                target_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', filename)
        else:
            target_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'nasa_real_battery_dataset.csv')
            if not os.path.exists(target_path):
                target_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'battery_aging_dataset.csv')

        if not os.path.exists(target_path):
            return jsonify({"success": False, "error": f"Dataset file not found: {target_path}"}), 404

        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        new_metadata = train_battery_models(target_path, models_dir)
        
        # Reload engine
        engine.load_models()
        
        return jsonify({
            "success": True,
            "message": f"Successfully retrained models on {os.path.basename(target_path)}!",
            "metrics": new_metadata['metrics'],
            "feature_importances": new_metadata['feature_importances']
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/api/analytics-data', methods=['GET'])
def api_analytics_data():
    try:
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'nasa_real_battery_dataset.csv')
        if not os.path.exists(data_path):
            data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'battery_aging_dataset.csv')
        if not os.path.exists(data_path):
            return jsonify({"success": False, "error": "Training dataset not found"}), 404

        df = pd.read_csv(data_path)
        
        # Select real NASA B0005 battery profile for trajectory plotting
        if 'battery_id' in df.columns and 'B0005' in df['battery_id'].values:
            b_sample = df[df['battery_id'] == 'B0005']
        elif 'battery_id' in df.columns:
            b_sample = df[df['battery_id'] == df['battery_id'].iloc[0]]
        else:
            b_sample = df.iloc[:300]
        
        trajectory = {
            "cycles": b_sample['cycle'].tolist(),
            "capacity": b_sample['capacity'].tolist(),
            "soh": b_sample['soh'].tolist(),
            "resistance": b_sample['resistance'].tolist(),
            "impedance": b_sample['impedance'].tolist(),
            "voltage": b_sample['voltage'].tolist(),
            "temperature": b_sample['temperature'].tolist()
        }
        
        # Correlation matrix for key features
        numeric_cols = ['voltage', 'current', 'temperature', 'capacity', 'cycle', 'resistance', 'impedance', 'soc', 'soh', 'rul']
        existing_cols = [c for c in numeric_cols if c in df.columns]
        corr_matrix = df[existing_cols].corr().round(3).to_dict()

        return jsonify({
            "success": True,
            "trajectory": trajectory,
            "correlations": corr_matrix,
            "metadata": engine.metadata
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "True").lower() in ("1", "true")
    print("\n========================================================")
    print(f"[INFO] EV Battery Life & Health Prediction System is Ready!")
    print(f"[INFO] Running on http://0.0.0.0:{port}")
    print("========================================================\n")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
