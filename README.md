# 🔋 VoltPredict AI: EV Battery Life & Health Prediction System

An end-to-end Machine Learning web dashboard designed for Electric Vehicle (EV) battery State of Health (**SoH**), Remaining Useful Life (**RUL** in cycles), and electrochemical degradation prognostics, tailored for **NASA PCoE** and **CALCE** battery datasets.

---

## ⚡ Supported Telemetry Parameters (All 11 Data Columns)

| Parameter | Unit | Description | Role in Degradation |
| :--- | :--- | :--- | :--- |
| `voltage` | Volts ($V$) | Cell terminal voltage under load | Voltage drop/sag indicates internal loss |
| `current` | Amperes ($A$) | Discharge / Charge current rate | High C-rates accelerate SEI layer growth |
| `temperature` | °C | Cell operating temperature | High heat (>45°C) & freezing (<5°C) accelerate degradation |
| `capacity` | Ampere-hours ($Ah$) | Discharge capacity | Primary metric for capacity retention |
| `cycle` | Count | Total cumulative charge-discharge cycles | Measure of cyclic battery aging |
| `time` | Seconds ($s$) | Elapsed operating or test duration | Temporal aging metric |
| `resistance` | Milliohms ($m\Omega$) | Internal ohmic cell resistance | Increases over cycles due to electrolyte loss |
| `impedance` | Ohms ($\Omega$) | Electrochemical impedance magnitude | EIS metric capturing charge-transfer resistance |
| `soc` | % | State of Charge (0 - 100%) | Operating depth of charge |
| `soh` | % | State of Health ($(C_{current} / C_{nominal}) \times 100$) | Key automotive health indicator |
| `age` | Days | Battery calendar age | Calendar aging index |

---

## 🚀 Quickstart Guide

### 1. Launch the Dashboard
From the project directory (`ev_battery_prediction`), run:
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### 2. What's Included

1. **Live Diagnostic Cockpit (`/`)**:
   - Interactive sliders and number inputs for all 11 battery parameters.
   - Quick-load scenario presets: **Brand New**, **Normal Aging (150c)**, **Second-Life (280c)**, and **Critical Thermal (360c)**.
   - Animated SVG SOH circular gauge.
   - Remaining Useful Life (**RUL in cycles**) and Estimated Remaining EV Driving Range (**km / miles**).
   - Dynamic Degradation Forecast plot to End of Life (**EOL = 70% SOH**).
   - Battery Management System (**BMS**) Smart Advisories (thermal alerts, impedance surge alerts, charge-rate advice).

2. **NASA & CALCE Dataset Laboratory (`/batch`)**:
   - Upload your own EDA-cleaned CSV datasets.
   - Automated schema inspection & column mapping.
   - First 10 rows data preview & summary statistics.
   - 1-Click ML Engine Retraining on uploaded data.
   - Pre-bundled sample dataset: `sample_nasa_b0005_test.csv`.

3. **EDA & Model Analytics (`/analytics`)**:
   - Capacity Fade degradation curve ($Ah$ vs Cycles).
   - Internal Resistance Growth ($m\Omega$ vs Cycles).
   - Feature Importance weights (Random Forest & Gradient Boosting).
   - Dual-axis temperature vs voltage sag evolution.

---

## 🛠️ Model Retraining with your EDA-Cleaned NASA / CALCE Datasets

Once you finish your Exploratory Data Analysis (EDA):
1. Make sure your final exported CSV includes the columns:
   `voltage, current, temperature, capacity, cycle, time, resistance, impedance, soc, soh`
2. You can either:
   - **Method A (Web UI)**: Go to `http://127.0.0.1:5000/batch`, drag & drop your CSV file, inspect the matched columns, and click **Retrain ML Engine**.
   - **Method B (Terminal)**: Save your CSV into `data/` and run:
     ```bash
     python ml_pipeline/train_model.py
     ```
3. The dashboard will automatically update its models and display fresh $R^2$, MAE, and RMSE metrics!
