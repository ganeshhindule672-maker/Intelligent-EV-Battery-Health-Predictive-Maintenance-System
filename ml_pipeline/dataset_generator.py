"""
ml_pipeline/dataset_generator.py

Generates realistic EV Battery degradation datasets modeled after
NASA Prognostics Center of Excellence (PCoE) and CALCE Battery research datasets.
Includes all requested columns:
  - voltage (V)
  - current (A)
  - temperature (°C)
  - capacity (Ah)
  - cycle (int)
  - time (s or hours)
  - resistance (mOhm or Ohm)
  - impedance (Ohm)
  - soc (%)
  - soh (%)
  - age (days)
  - rul (Remaining Useful Life in cycles)
"""

import os
import numpy as np
import pandas as pd

def generate_battery_degradation_data(
    num_batteries=15,
    cycles_per_battery=350,
    nominal_capacity=2.0,  # 2.0 Ah standard 18650 cell
    eol_ratio=0.70,        # End of Life threshold at 70% of nominal capacity (1.4 Ah)
    random_seed=42
):
    np.random.seed(random_seed)
    records = []
    
    eol_capacity = nominal_capacity * eol_ratio

    for b_id in range(1, num_batteries + 1):
        # Battery-specific degradation rates and initial conditions
        # Different ambient conditions (e.g., elevated temp degrades faster)
        base_temp = np.random.choice([24.0, 30.0, 40.0, 45.0], p=[0.4, 0.3, 0.2, 0.1])
        temp_factor = 1.0 + (base_temp - 24.0) * 0.015
        
        # Degradation acceleration factor
        deg_rate = np.random.uniform(0.0006, 0.0011) * temp_factor
        initial_resistance = np.random.uniform(0.045, 0.060) # Ohms
        initial_impedance = np.random.uniform(0.035, 0.050)  # Ohms
        
        # Simulate life until EOL or max cycles
        current_capacity = nominal_capacity * np.random.uniform(0.98, 1.02)
        total_life_cycles = int(np.random.uniform(280, 420) / (temp_factor * 0.95))
        
        cumulative_time = 0.0
        
        for cycle in range(1, total_life_cycles + 1):
            # Calculate degradation trajectory with electrochemical aging
            # Power law capacity fade with noise and occasional capacity regeneration
            regen = 0.003 if (cycle % 25 == 0) else 0.0
            noise = np.random.normal(0, 0.003)
            
            # Decay curve
            capacity_fade = (deg_rate * (cycle ** 1.05)) - regen
            cap = nominal_capacity - capacity_fade + noise
            cap = min(nominal_capacity * 1.02, max(cap, 0.9)) # clamp
            
            # SOH (%)
            soh = (cap / nominal_capacity) * 100.0
            
            # RUL (Remaining Useful Life in cycles)
            rul = max(0, total_life_cycles - cycle)
            
            # Resistance growth (SEI layer formation and electrolyte depletion)
            resistance_growth = initial_resistance * (1.0 + 0.0028 * cycle + 0.000003 * (cycle ** 2))
            res_noise = np.random.normal(0, 0.001)
            resistance_ohm = max(0.04, resistance_growth + res_noise)
            resistance_mohm = resistance_ohm * 1000.0 # mOhm
            
            # Impedance (Electrochemical Impedance Spectroscopy magnitude)
            impedance = initial_impedance * (1.0 + 0.0023 * cycle) + np.random.normal(0, 0.001)
            
            # Operating conditions during cycle
            # Nominal operating voltage drops as battery ages and resistance climbs
            soc = np.random.uniform(15.0, 95.0) # current operating SoC
            ocv = 3.2 + (soc / 100.0) * 0.95    # Open circuit voltage baseline (3.2V to 4.15V)
            
            current_draw = np.random.uniform(1.0, 2.5) # Discharge current (Amps)
            # Operating voltage = OCV - I * R
            voltage = ocv - (current_draw * resistance_ohm) + np.random.normal(0, 0.01)
            voltage = np.clip(voltage, 2.7, 4.25)
            
            # Operating temperature rise due to Joule heating: Q = I^2 * R
            heat_gen = (current_draw ** 2) * resistance_ohm * 4.0
            operating_temp = base_temp + heat_gen + np.random.normal(0, 0.8)
            
            # Time per cycle (approx 3600 - 4500 seconds)
            cycle_duration = np.random.uniform(3400, 4200) # seconds
            cumulative_time += cycle_duration
            
            # Age in days (assuming ~1.5 cycles per day on average)
            age_days = round(cycle * 0.75 + np.random.uniform(0, 2), 1)

            records.append({
                'battery_id': f'B{b_id:04d}',
                'cycle': int(cycle),
                'voltage': round(float(voltage), 4),
                'current': round(float(current_draw), 3),
                'temperature': round(float(operating_temp), 2),
                'capacity': round(float(cap), 4),
                'time': round(float(cumulative_time), 1),
                'resistance': round(float(resistance_mohm), 2),     # in mOhm
                'impedance': round(float(impedance), 4),            # in Ohm
                'soc': round(float(soc), 2),                        # %
                'soh': round(float(soh), 2),                        # %
                'age': round(float(age_days), 1),                   # days
                'rul': int(rul)                                     # remaining cycles
            })

    df = pd.DataFrame(records)
    return df

def save_default_datasets(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating comprehensive NASA/CALCE battery aging training dataset...")
    df_full = generate_battery_degradation_data(num_batteries=16, cycles_per_battery=350, random_seed=42)
    full_path = os.path.join(output_dir, "battery_aging_dataset.csv")
    df_full.to_csv(full_path, index=False)
    print(f"-> Saved training dataset ({len(df_full)} rows) to: {full_path}")

    # Also generate a smaller sample dataset representing NASA B0005 cell for testing user uploads
    sample_df = df_full[df_full['battery_id'] == 'B0001'].copy()
    sample_path = os.path.join(output_dir, "sample_nasa_b0005_test.csv")
    sample_df.to_csv(sample_path, index=False)
    print(f"-> Saved test upload sample dataset ({len(sample_df)} rows) to: {sample_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    save_default_datasets(os.path.join(project_root, "data"))
