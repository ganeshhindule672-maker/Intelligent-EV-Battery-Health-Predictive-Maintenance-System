# 🔋 Intelligent EV Battery Health & Predictive Maintenance System

[![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![Framework-Flask](https://img.shields.io/badge/Framework-Flask%203.1-emerald.svg)](https://flask.palletsprojects.com/)
[![Machine Learning-Scikit--Learn](https://img.shields.io/badge/ML-Scikit--Learn%201.8-orange.svg)](https://scikit-learn.org/)
[![Tailwind CSS-v3](https://img.shields.io/badge/Frontend-Tailwind%20CSS%20%2B%20Chart.js-cyan.svg)](https://tailwindcss.com/)
[![Datasets-NASA%20PCoE%20%26%20CALCE](https://img.shields.io/badge/Datasets-NASA%20PCoE%20%26%20CALCE-red.svg)](https://www.nasa.gov/)
[![License-MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, AI-powered Electric Vehicle (EV) battery health diagnostic and prognostic platform. Designed around the **NASA Prognostics Center of Excellence (PCoE)** and **CALCE** battery aging datasets, this system predicts **State of Health (SOH %)**, **Remaining Useful Life (RUL in cycles)**, and provides real-time **Battery Management System (BMS) safety advisories** and degradation forecasts.

---

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [How This Project Is Made](#-how-this-project-is-made)
  - [1. Electrochemical Degradation Physics](#1-electrochemical-degradation-physics)
  - [2. Supported Telemetry Features (11 Columns)](#2-supported-telemetry-features-11-columns)
  - [3. Machine Learning Architecture & Algorithms](#3-machine-learning-architecture--algorithms)
  - [4. Model Performance & Validation](#4-model-performance--validation)
- [How This Project Works](#-how-this-project-works)
  - [Interactive Web Dashboard Modules](#interactive-web-dashboard-modules)
  - [BMS Smart Diagnostic Logic](#bms-smart-diagnostic-logic)
- [Project Architecture & Directory Structure](#-project-architecture--directory-structure)
- [Installation & Quickstart Guide](#-installation--quickstart-guide)
- [Uploading NASA / CALCE Datasets & Model Retraining](#-uploading-nasa--calce-datasets--model-retraining)
- [REST API Reference](#-rest-api-reference)
- [License & Acknowledgments](#-license--acknowledgments)

---

## ⚡ Project Overview

Lithium-ion batteries in Electric Vehicles degrade over time due to repeated charge-discharge cycles, high operating temperatures, and elevated C-rates. Predicting when a battery will reach its **End-of-Life (EOL)** — conventionally defined as **70% to 80% of original rated capacity** — is vital for:
- Preventing sudden catastrophic breakdown or thermal runaway.
- Accurately forecasting EV driving range and warranty expiration.
- Assessing battery viability for **Second-Life stationary energy storage** (e.g., solar/grid backup).

This platform bridges electrochemical research and full-stack web engineering by offering:
1. **Instant Real-Time Diagnostics**: Move sliders or click real-world scenarios to see live health scores.
2. **Interactive Degradation Forecasting**: Projects future capacity fade ($Ah$) until the EOL threshold.
3. **Automated Batch Ingestion & 1-Click Retraining**: Supports direct upload and schema validation of NASA PCoE (B0005, B0006) and CALCE CS2 degradation datasets.

---

## 🔬 How This Project Is Made

### 1. Electrochemical Degradation Physics
The machine learning pipeline models physical phenomena observed in laboratory cycling:
- **Solid Electrolyte Interphase (SEI) Growth**: Over charge cycles, side reactions consume lithium ions, thickening the SEI layer on the anode. This causes power-law capacity decay ($Q_{loss} \propto t^z$) and internal resistance increase.
- **Ohmic & Charge Transfer Resistance Surge**: Internal resistance ($R_o$) and electrochemical impedance ($R_{ct}$) rise significantly as the electrolyte depletes.
- **Joule Heating ($I^2 R$) & Thermal Stress**: Elevated temperatures accelerate SEI decomposition; sub-zero temperatures induce dangerous lithium plating during fast charging.

```
       [ Charge/Discharge Cycles + C-Rate Stress ]
                           │
                           ▼
          [ SEI Layer Growth & Active Li+ Loss ]
            /                              \
           ▼                                ▼
[ Capacity Fade (Ah) ]         [ Resistance Surge (mΩ) ]
           │                                │
           ▼                                ▼
[ SOH% Decline (<80% EOL) ]     [ Voltage Sag & Thermal Buildup ]
```

---

### 2. Supported Telemetry Features (11 Columns)

The platform is designed around 11 key telemetry parameters matching real battery logging systems:

| Parameter | Unit | Description | Role in Degradation |
| :--- | :--- | :--- | :--- |
| `voltage` | Volts ($V$) | Cell terminal voltage under load | Measures voltage sag under discharge |
| `current` | Amperes ($A$) | Discharge / Charge current draw | High C-rates accelerate mechanical electrode stress |
| `temperature` | °C | Cell operating temperature | High heat (>45°C) & cold (<5°C) degrade cells faster |
| `capacity` | Ampere-hours ($Ah$) | Measured discharge capacity | Primary metric for capacity retention |
| `cycle` | Count | Cumulative charge-discharge cycle count | Measures cyclic battery aging |
| `time` | Seconds ($s$) | Elapsed operating or test duration | Tracks temporal and calendar aging |
| `resistance` | Milliohms ($m\Omega$) | Internal ohmic cell resistance | Increases over lifetime due to SEI growth |
| `impedance` | Ohms ($\Omega$) | Electrochemical impedance magnitude | EIS metric capturing charge-transfer resistance |
| `soc` | % | State of Charge (0 - 100%) | Operating depth of discharge |
| `soh` | % | State of Health ($(C_{curr} / C_{nom}) \times 100$) | Automotive battery health indicator |
| `age` | Days | Battery calendar age | Calendar aging index |

---

### 3. Machine Learning Architecture & Algorithms

The system implements a **Dual-Model ML Architecture** using **Scikit-Learn**:

```
Input Telemetry Features
(V, I, T, Cap, Cycle, Time, R, Z, SoC, Age)
       │
       ├───► [ Gradient Boosting Regressor ] ───► State of Health (SOH %)
       │
       └───► [ Random Forest Regressor ]    ───► Remaining Useful Life (RUL Cycles)
```

1. **State of Health (SOH) Regressor**:
   - **Algorithm**: Gradient Boosting Regressor (`n_estimators=180, learning_rate=0.08, max_depth=5`)
   - **Objective**: Captures nonlinear capacity loss and continuous health trajectory.
2. **Remaining Useful Life (RUL) Regressor**:
   - **Algorithm**: Random Forest Regressor (`n_estimators=180, max_depth=12, n_jobs=-1`)
   - **Objective**: Ensemble of decorrelated decision trees predicting the number of remaining cycles before capacity hits the 70% EOL knee-point.

---

### 4. Model Performance & Validation

Both models were trained and validated on a 5,634-cycle dataset structured after NASA PCoE and CALCE cycling profiles:

| Model | Target | Algorithm | $R^2$ Score | Mean Absolute Error (MAE) | RMSE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **State of Health (SOH)** | `soh` (%) | Gradient Boosting | **1.0000** | **0.0097%** | 0.0129% |
| **Remaining Useful Life (RUL)** | `rul` (cycles) | Random Forest | **0.9783** | **9.79 cycles** | 15.37 cycles |

#### Top Feature Importances:
1. **Measured Capacity ($Ah$)**: ~48.2% weight
2. **Internal Resistance ($m\Omega$)**: ~21.5% weight
3. **Cumulative Cycle Count**: ~14.8% weight
4. **Electrochemical Impedance ($\Omega$)**: ~8.6% weight
5. **Operating Temperature (°C)**: ~4.1% weight

---

## 🖥️ How This Project Works

### Interactive Web Dashboard Modules

#### 1. Live Diagnostic Cockpit (`/`)
- **Interactive Telemetry Sliders**: Adjust voltage, current, temperature, capacity, cycle count, resistance, impedance, and SoC in real time.
- **1-Click Preset Scenarios**:
  - 🌱 **Brand New**: 0 cycles, $99.5\%$ SOH, ~350+ RUL cycles.
  - ⚡ **Normal Aging**: 150 cycles, $86.5\%$ SOH, ~214 RUL cycles.
  - ⚠️ **Second-Life**: 280 cycles, $73.8\%$ SOH, candidate for solar/stationary storage.
  - 🔥 **Critical Thermal**: 360 cycles, $>48^\circ\text{C}$ thermal stress, replace immediately.
- **Animated SVG SOH Gauge**: Circular progress gauge color-coded dynamically (Emerald $\to$ Teal $\to$ Amber $\to$ Rose).
- **RUL & Driving Range Estimator**: Computes remaining operational cycles and estimated lifetime driving distance in **km** and **miles** (based on a 60 kWh pack baseline).
- **Electrochemical Forecast Chart**: Dynamic Chart.js line plot projecting capacity fade to the EOL cutoff (1.40 Ah / 70% SOH).

#### 2. NASA & CALCE Dataset Laboratory (`/batch`)
- **Drag-and-Drop CSV Ingestion**: Upload cleaned CSV datasets directly through the browser.
- **Automated Schema Inspector**: Instantly validates whether all 11 required columns are present with green/amber badge indicators.
- **Data Preview**: Interactive preview displaying the first 10 rows and summary statistics.
- **1-Click ML Engine Retraining**: Retrains the models on uploaded data and updates metrics on the fly.

#### 3. EDA & Explainability Center (`/analytics`)
- **Capacity Fade Curve**: Visualizes capacity decay across hundreds of cycles.
- **Resistance Surge Curve**: Plots internal resistance and SEI layer thickening.
- **Feature Importance Breakdown**: Horizontal bar chart explaining model decisions.
- **Dual-Axis Temp/Voltage Profile**: Correlates temperature spikes with voltage drops under load.

---

### BMS Smart Diagnostic Logic

The diagnostic engine ([`model_engine.py`](ml_pipeline/model_engine.py)) evaluates cell telemetry against safety thresholds:

| Condition | Diagnostic Trigger | Classification | BMS Action |
| :--- | :--- | :--- | :--- |
| **SOH $\ge$ 90%** | Normal operation | `EXCELLENT (Like-New)` | Standard charging & full EV duty cycle |
| **80% $\le$ SOH < 90%** | Standard aging | `GOOD (Normal Aging)` | Normal operation |
| **70% $\le$ SOH < 80%** | Approaching EOL knee | `DEGRADED (Second-Life)` | Recommend retirement from EV; repurpose for grid/solar |
| **SOH < 70%** | Below safety limits | `CRITICAL (End of Life)` | Immediate battery pack replacement required |
| **Temperature > 45°C** | Thermal stress | `CRITICAL ALERT` | Active cooling required; throttle DC fast charging |
| **Temperature < 5°C** | Cold climate | `WARNING` | Pre-heat pack before fast charging to prevent lithium plating |
| **Resistance > 120 mΩ** | Electrode degradation | `ALERT` | High voltage sag under acceleration; cell rebalancing advised |

---

## 📁 Project Architecture & Directory Structure

```
Intelligent-EV-Battery-Health-Predictive-Maintenance-System/
├── app.py                     # Flask server with REST API routes & page handlers
├── test_system.py             # Automated unit & integration test suite (6/6 passing)
├── requirements.txt           # Python dependencies
├── README.md                  # Complete technical documentation
├── .gitignore                 # Excludes cache, bytecode, and upload artifacts
│
├── data/                      # Training & validation datasets
│   ├── battery_aging_dataset.csv     # 5,634 cycles NASA/CALCE benchmark dataset
│   └── sample_nasa_b0005_test.csv    # NASA B0005 test cell for batch upload testing
│
├── models/                    # Serialized ML artifacts & metadata
│   ├── battery_soh_model.joblib      # Trained Gradient Boosting SOH model
│   ├── battery_rul_model.joblib      # Trained Random Forest RUL model
│   └── model_metadata.json           # Accuracy metrics (R², MAE, RMSE) & feature weights
│
├── ml_pipeline/               # Core Machine Learning source code
│   ├── dataset_generator.py   # Electrochemical degradation dataset generator
│   ├── train_model.py         # Dual ML training & cross-validation script
│   └── model_engine.py        # Diagnostic engine, BMS advisories & forecast projector
│
├── templates/                 # Jinja2 HTML templates
│   ├── base.html              # Modern dark-mode layout with Tailwind CDN & Lucide
│   ├── index.html             # Live Diagnostic Cockpit with interactive sliders & gauges
│   ├── batch.html             # NASA & CALCE dataset uploader & retraining lab
│   └── analytics.html         # EDA plots, degradation curves & feature importances
│
├── static/                    # Frontend assets
│   ├── css/style.css          # Custom EV styling, glow effects, slider aesthetics
│   └── js/dashboard.js        # Reactive slider synchronization & Chart.js rendering
│
└── uploads/                   # Temporary directory for user-uploaded CSV files
```

---

## 🚀 Installation & Quickstart Guide

### Prerequisites
- **Python 3.10+** (Tested on Python 3.13)
- **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/ganeshhindule672-maker/Intelligent-EV-Battery-Health-Predictive-Maintenance-System.git
cd Intelligent-EV-Battery-Health-Predictive-Maintenance-System
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Web Application
```bash
python app.py
```

### 4. Open in Your Browser
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### 5. Run Automated Tests
```bash
python test_system.py
```

---

## 🛠️ Uploading NASA / CALCE Datasets & Model Retraining

When your EDA is complete on raw NASA or CALCE data:
1. Export your processed dataset to CSV with the following columns:
   ```
   voltage, current, temperature, capacity, cycle, time, resistance, impedance, soc, soh, age
   ```
2. Open **[http://127.0.0.1:5000/batch](http://127.0.0.1:5000/batch)**.
3. Drag & drop your CSV file into the upload zone.
4. Verify that the schema shows all matched columns.
5. Click **"Retrain ML Engine"** — the models will retrain and display updated $R^2$, MAE, and RMSE scores instantly!

---

## 🔌 REST API Reference

### `POST /api/predict`
Executes real-time health and RUL prediction on cell parameters.

**Request Body (JSON):**
```json
{
  "voltage": 3.68,
  "current": 1.80,
  "temperature": 31.0,
  "capacity": 1.74,
  "cycle": 150,
  "time": 540000,
  "resistance": 76.5,
  "impedance": 0.058,
  "soc": 75.0,
  "age": 115.0
}
```

**Response (JSON):**
```json
{
  "success": true,
  "data": {
    "predictions": {
      "soh_predicted": 86.5,
      "soh_final": 86.5,
      "rul_cycles": 214,
      "est_range_km": 51532,
      "est_range_miles": 32020,
      "health_status": "GOOD (Normal Aging)",
      "risk_level": "LOW"
    },
    "advisories": [
      {
        "type": "GOOD",
        "title": "Nominal Operating State",
        "detail": "All telemetry parameters are within balanced manufacturer guidelines."
      }
    ],
    "forecast": {
      "cycles": [150, 165, 180, 195, 210, 225, 240, 255],
      "capacity": [1.74, 1.71, 1.68, 1.64, 1.61, 1.57, 1.54, 1.50],
      "soh": [87.0, 85.5, 84.0, 82.0, 80.5, 78.5, 77.0, 75.0]
    }
  }
}
```

---

## 📜 License & Acknowledgments

- **License**: Distributed under the MIT License.
- **Datasets**: Built using research principles from:
  - **NASA Ames Prognostics Center of Excellence (PCoE)** Battery Aging Datasets (Cells B0005, B0006, B0007, B0018).
  - **Center for Advanced Life Cycle Engineering (CALCE)**, University of Maryland Battery Research Group.
