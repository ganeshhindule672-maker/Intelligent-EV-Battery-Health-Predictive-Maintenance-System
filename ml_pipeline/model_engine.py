"""
ml_pipeline/model_engine.py

Model inference and Battery Health Diagnostic Engine.
Provides single-sample and batch prediction, health status classification,
electrochemical risk factors, BMS advisories, and degradation forecasting.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

class BatteryHealthEngine:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.soh_model = None
        self.rul_model = None
        self.metadata = {}
        self.load_models()

    def load_models(self):
        soh_path = os.path.join(self.models_dir, "battery_soh_model.joblib")
        rul_path = os.path.join(self.models_dir, "battery_rul_model.joblib")
        meta_path = os.path.join(self.models_dir, "model_metadata.json")

        if os.path.exists(soh_path):
            self.soh_model = joblib.load(soh_path)
        if os.path.exists(rul_path):
            self.rul_model = joblib.load(rul_path)
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)

    @property
    def is_ready(self):
        return self.soh_model is not None and self.rul_model is not None

    def predict_single(self, params: dict):
        """
        params: dict with keys:
          voltage, current, temperature, capacity, cycle,
          time, resistance, impedance, soc, soh (optional), age (optional)
        """
        # Default fallbacks if keys missing
        cycle = float(params.get('cycle', 100))
        voltage = float(params.get('voltage', 3.7))
        current = float(params.get('current', 1.5))
        temp = float(params.get('temperature', 25.0))
        capacity = float(params.get('capacity', 1.85))
        time_val = float(params.get('time', cycle * 3600.0))
        resistance = float(params.get('resistance', 65.0)) # mOhm
        impedance = float(params.get('impedance', 0.05))   # Ohm
        soc = float(params.get('soc', 80.0))
        age = float(params.get('age', cycle * 0.75))

        # 1. Predict SoH if not provided or to compare
        soh_features = ['voltage', 'current', 'temperature', 'capacity', 'cycle', 'time', 'resistance', 'impedance', 'soc', 'age']
        df_soh = pd.DataFrame([[voltage, current, temp, capacity, cycle, time_val, resistance, impedance, soc, age]], columns=soh_features)
        
        if self.soh_model:
            predicted_soh = float(self.soh_model.predict(df_soh)[0])
        else:
            # Physics-based baseline approximation
            predicted_soh = max(20.0, min(100.0, (capacity / 2.0) * 100.0))

        # If user explicitly supplied SoH and it's valid, respect user's calculated SoH or blend
        user_soh = params.get('soh')
        final_soh = float(user_soh) if (user_soh is not None and str(user_soh).strip() != '') else predicted_soh
        final_soh = max(10.0, min(100.0, final_soh))

        # 2. Predict RUL
        rul_features = ['voltage', 'current', 'temperature', 'capacity', 'cycle', 'time', 'resistance', 'impedance', 'soc', 'soh', 'age']
        df_rul = pd.DataFrame([[voltage, current, temp, capacity, cycle, time_val, resistance, impedance, soc, final_soh, age]], columns=rul_features)
        if self.rul_model:
            predicted_rul = max(0, int(round(self.rul_model.predict(df_rul)[0])))
        else:
            # Heuristic calculation if model not ready
            remaining_cycles_est = max(0, int((final_soh - 70.0) / 0.08))
            predicted_rul = remaining_cycles_est

        # 3. Health Classification
        if final_soh >= 90.0:
            health_status = "EXCELLENT (Like-New)"
            status_color = "emerald"
            badge_class = "badge-optimal"
            summary_desc = "Battery is in prime electrochemical condition with minimal capacity loss."
        elif final_soh >= 80.0:
            health_status = "GOOD (Normal Aging)"
            status_color = "teal"
            badge_class = "badge-good"
            summary_desc = "Normal cyclic degradation observed. Meets all high-performance EV specifications."
        elif final_soh >= 70.0:
            health_status = "DEGRADED (Second-Life Candidate)"
            status_color = "amber"
            badge_class = "badge-warning"
            summary_desc = "Battery capacity is approaching EV retirement threshold (70-80% SOH). Ideal for stationary energy storage."
        else:
            health_status = "CRITICAL (End of Life - Replace)"
            status_color = "rose"
            badge_class = "badge-critical"
            summary_desc = "Battery has exceeded safe automotive operational limits. High internal resistance and risk of rapid discharge."

        # 4. Range Estimation & Longevity in Human Terms
        km_per_cycle = 280.0
        est_remaining_km = int(predicted_rul * km_per_cycle * (final_soh / 100.0))
        est_remaining_miles = int(est_remaining_km * 0.621371)
        est_years_remaining = round(max(0.0, predicted_rul / 180.0), 1)

        # Replacement urgency
        if final_soh >= 85.0:
            urgency = "Minimal (Long Lifespan Remaining)"
            urgency_color = "emerald"
        elif final_soh >= 75.0:
            urgency = "Normal Aging (~1-2 Years Left)"
            urgency_color = "teal"
        elif final_soh >= 70.0:
            urgency = "Moderate (Plan Replacement in < 6-12 Months)"
            urgency_color = "amber"
        else:
            urgency = "High (Immediate Replacement / Second-Life Candidate)"
            urgency_color = "rose"

        # Impact breakdown (Explains what is driving battery degradation)
        capacity_loss_pct = round(max(0.0, (2.0 - capacity) / 2.0 * 100.0), 1)
        cycle_wear_level = "Low" if cycle < 100 else ("Moderate" if cycle < 250 else "High")
        
        if temp > 40:
            thermal_impact = "Severe Heat Stress (Accelerating aging by ~50%)"
            thermal_score = "Severe"
        elif temp > 32:
            thermal_impact = "Elevated Temperature (Accelerating aging by ~15%)"
            thermal_score = "Warm"
        elif temp < 10:
            thermal_impact = "Cold Ambient (Reduced instantaneous range, sluggish ions)"
            thermal_score = "Cold"
        else:
            thermal_impact = "Optimal Operating Temp (Minimal thermal penalty)"
            thermal_score = "Optimal"

        res_growth_pct = round(max(0.0, (resistance - 48.0) / 48.0 * 100.0), 1)
        res_health = "Optimal (<60 mΩ)" if resistance < 60 else ("Normal Aging (60-90 mΩ)" if resistance < 90 else ("High Resistance (90-130 mΩ)" if resistance < 130 else "Severe Degradation (>130 mΩ)"))

        impacts = {
            "capacity_loss_pct": capacity_loss_pct,
            "cycle_wear_level": cycle_wear_level,
            "thermal_impact": thermal_impact,
            "thermal_score": thermal_score,
            "resistance_growth_pct": res_growth_pct,
            "resistance_health": res_health,
            "est_years_remaining": est_years_remaining,
            "urgency": urgency,
            "urgency_color": urgency_color
        }

        # 5. Risk Factors & BMS Alerts
        advisories = []
        risk_level = "LOW"
        
        # Thermal check
        if temp > 45.0:
            advisories.append({
                "type": "CRITICAL",
                "title": "Severe Thermal Stress (>45°C)",
                "detail": "Elevated cell temperature accelerates Solid Electrolyte Interphase (SEI) growth and cathode dissolution. BMS thermal cooling recommended."
            })
            risk_level = "HIGH"
        elif temp > 35.0:
            advisories.append({
                "type": "WARNING",
                "title": "Moderate Heat Buildup (35-45°C)",
                "detail": "Ensure cooling system circulation is active during high-power discharge or DC fast charging."
            })
            if risk_level != "HIGH": risk_level = "MODERATE"
        elif temp < 5.0:
            advisories.append({
                "type": "WARNING",
                "title": "Sub-zero / Cold Ambient Warning (<5°C)",
                "detail": "Pre-heat battery pack before DC fast charging to prevent irreversible lithium metal plating on anode."
            })

        # Internal Resistance check
        if resistance > 120.0 or impedance > 0.12:
            advisories.append({
                "type": "ALERT",
                "title": "High Internal Impedance",
                "detail": f"Measured resistance ({resistance} mΩ) is significantly elevated, causing increased voltage sag under acceleration."
            })
            if risk_level != "HIGH": risk_level = "MODERATE"

        # State of Charge check
        if soc > 90.0:
            advisories.append({
                "type": "INFO",
                "title": "High SoC Stress",
                "detail": "Prolonged storage at >90% SoC at high temperature increases chemical stress. Limit daily charging to 80%."
            })
        elif soc < 15.0:
            advisories.append({
                "type": "INFO",
                "title": "Deep Discharge Warning",
                "detail": "Recharge soon to prevent copper dissolution and internal short-circuit hazards."
            })

        if not advisories:
            advisories.append({
                "type": "GOOD",
                "title": "Nominal Operating State",
                "detail": "All telemetry parameters are within balanced manufacturer guidelines."
            })

        # 6. Projected Degradation Curve (from current cycle to EOL)
        forecast_cycles = []
        forecast_capacity = []
        forecast_soh = []
        
        curr_c = int(cycle)
        end_c = curr_c + predicted_rul + 50
        step = max(5, int((end_c - curr_c) / 15))
        
        nominal_cap = 2.0
        cap_drop_per_cycle = (capacity - 1.4) / max(1, predicted_rul) if predicted_rul > 0 else 0.002
        
        for c in range(curr_c, end_c + step, step):
            delta = c - curr_c
            proj_cap = max(1.1, capacity - (delta * cap_drop_per_cycle))
            proj_soh = max(50.0, (proj_cap / nominal_cap) * 100.0)
            forecast_cycles.append(c)
            forecast_capacity.append(round(proj_cap, 3))
            forecast_soh.append(round(proj_soh, 2))

        return {
            "inputs": {
                "voltage": voltage,
                "current": current,
                "temperature": temp,
                "capacity": capacity,
                "cycle": int(cycle),
                "time": time_val,
                "resistance": resistance,
                "impedance": impedance,
                "soc": soc,
                "soh": final_soh,
                "age": age
            },
            "predictions": {
                "soh_predicted": round(predicted_soh, 2),
                "soh_final": round(final_soh, 2),
                "rul_cycles": predicted_rul,
                "est_range_km": est_remaining_km,
                "est_range_miles": est_remaining_miles,
                "est_years_remaining": est_years_remaining,
                "health_status": health_status,
                "status_color": status_color,
                "badge_class": badge_class,
                "summary_desc": summary_desc,
                "risk_level": risk_level,
                "urgency": urgency,
                "urgency_color": urgency_color
            },
            "impacts": impacts,
            "advisories": advisories,
            "forecast": {
                "cycles": forecast_cycles,
                "capacity": forecast_capacity,
                "soh": forecast_soh
            }
        }

    def get_analytics_summary(self):
        return self.metadata
