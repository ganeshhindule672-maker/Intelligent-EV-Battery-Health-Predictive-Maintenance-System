"""
ml_pipeline/process_user_data.py

Processes the user's authentic NASA battery dataset (cleaned_MERGED_DATA_SET.xlsx)
into the standard 11-column training dataset with RUL labels.
"""

import os
import numpy as np
import pandas as pd

def process_nasa_dataset(
    excel_path=r"C:\Users\Ganesh\OneDrive\Desktop\Project\data for project\cleaned_MERGED_DATA_SET.xlsx",
    output_csv="data/nasa_real_battery_dataset.csv"
):
    if not os.path.exists(excel_path):
        # Fallback to Downloads folder
        alt_path = r"C:\Users\Ganesh\Downloads\cleaned_MERGED_DATA_SET.xlsx"
        if os.path.exists(alt_path):
            excel_path = alt_path
        else:
            raise FileNotFoundError(f"Could not find NASA dataset at {excel_path} or {alt_path}")

    print(f"Loading user dataset from: {excel_path}...")
    df = pd.read_excel(excel_path)
    print(f"Loaded raw dataset with {len(df)} rows.")

    # 1. Extract and clean impedance lookup (Re and Rct)
    imp_df = df[df['type'] == 'impedance'][['battery_id', 'test_id', 'Re', 'Rct']].dropna()
    # Filter non-physical negative or extreme values
    imp_df = imp_df[(imp_df['Re'] > 0.01) & (imp_df['Re'] < 1.0) & (imp_df['Rct'] > 0.01) & (imp_df['Rct'] < 2.0)]
    imp_df = imp_df.sort_values(['battery_id', 'test_id'])

    # 2. Extract discharge cycles with realistic capacity
    dis_df = df[(df['type'] == 'discharge') & (df['Capacity'] > 0.5)].copy()
    dis_df = dis_df.sort_values(['battery_id', 'test_id'])

    # 3. Merge impedance by nearest/previous test_id per battery
    merged_rows = []
    for b_id, grp in dis_df.groupby('battery_id'):
        b_imp = imp_df[imp_df['battery_id'] == b_id]
        grp_clean = grp.drop(columns=['Re', 'Rct'], errors='ignore')
        if not b_imp.empty:
            merged = pd.merge_asof(
                grp_clean.sort_values('test_id'),
                b_imp[['test_id', 'Re', 'Rct']].sort_values('test_id'),
                on='test_id',
                direction='nearest'
            )
        else:
            merged = grp_clean.copy()
            merged['Re'] = np.nan
            merged['Rct'] = np.nan
        merged_rows.append(merged)

    data = pd.concat(merged_rows, ignore_index=True)

    # 4. Fill missing Re/Rct with physically realistic defaults based on cycle progression
    data['cycle'] = data.groupby('battery_id').cumcount() + 1
    
    # Resistance in mOhm: (Re + Rct) * 1000
    base_res = 50.0 + data['cycle'] * 0.35 + np.random.normal(0, 1.5, size=len(data))
    measured_res = (data['Re'] + data['Rct']) * 1000.0
    data['resistance'] = np.where(measured_res.notnull() & (measured_res > 20) & (measured_res < 350), measured_res, base_res)
    data['resistance'] = np.clip(data['resistance'], 35.0, 350.0)

    # Impedance in Ohm
    base_imp = 0.035 + data['cycle'] * 0.00025 + np.random.normal(0, 0.001, size=len(data))
    measured_imp = np.sqrt(data['Re']**2 + data['Rct']**2)
    data['impedance'] = np.where(measured_imp.notnull() & (measured_imp > 0.01) & (measured_imp < 0.5), measured_imp, base_imp)
    data['impedance'] = np.clip(data['impedance'], 0.02, 0.45)

    # 5. Clean Voltage, Current, Temperature
    data['voltage'] = np.clip(data['Voltage_measured'].fillna(3.7), 2.5, 4.35)
    data['current'] = np.clip(np.abs(data['Current_measured'].fillna(1.5)), 0.1, 5.0)
    
    # NASA dataset: ambient_temperature + Temperature_measured
    amb = data['ambient_temperature'].fillna(24.0)
    delta_t = data['Temperature_measured'].fillna(8.0)
    data['temperature'] = np.clip(amb + delta_t, 10.0, 65.0)

    data['capacity'] = data['Capacity'].round(4)
    data['time'] = data['Time'].fillna(data['cycle'] * 3600.0).round(1)

    # 6. SOH (%) and RUL (Cycles)
    # Group by battery and compute initial capacity (or 2.0 Ah nominal standard)
    initial_caps = data.groupby('battery_id')['capacity'].transform('first')
    nominal_cap = np.where(initial_caps > 1.4, initial_caps, 2.0)
    
    data['soh'] = np.clip((data['capacity'] / nominal_cap) * 100.0, 30.0, 100.0).round(2)
    data['soc'] = np.clip(((data['voltage'] - 2.8) / (4.2 - 2.8)) * 100.0, 10.0, 98.0).round(2)
    data['age'] = (data['cycle'] * 0.75 + np.random.uniform(0, 2, size=len(data))).round(1)

    # Compute RUL: remaining cycles before capacity hits 1.40 Ah (70% threshold)
    rul_list = []
    for b_id, grp in data.groupby('battery_id'):
        eol_candidates = grp[grp['capacity'] <= 1.40]
        if not eol_candidates.empty:
            eol_cycle = eol_candidates['cycle'].min()
        else:
            eol_cycle = grp['cycle'].max()
        ruls = np.maximum(0, eol_cycle - grp['cycle'])
        grp_copy = grp.copy()
        grp_copy['rul'] = ruls.astype(int)
        rul_list.append(grp_copy)

    final_df = pd.concat(rul_list, ignore_index=True)

    # 7. Select and round the 11 target columns + metadata
    target_cols = [
        'battery_id', 'cycle', 'voltage', 'current', 'temperature', 
        'capacity', 'time', 'resistance', 'impedance', 'soc', 'soh', 'age', 'rul'
    ]
    final_df = final_df[target_cols]

    # Clean rounding
    final_df['voltage'] = final_df['voltage'].round(4)
    final_df['current'] = final_df['current'].round(3)
    final_df['temperature'] = final_df['temperature'].round(2)
    final_df['resistance'] = final_df['resistance'].round(2)
    final_df['impedance'] = final_df['impedance'].round(4)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    print(f"\n-> Successfully processed {len(final_df)} real NASA battery degradation cycles across {final_df['battery_id'].nunique()} batteries!")
    print(f"-> Saved to: {output_csv}")

    # Also save a sample for B0005 for quick inspection
    b5 = final_df[final_df['battery_id'] == 'B0005']
    b5_path = os.path.join(os.path.dirname(output_csv), "sample_real_nasa_b0005.csv")
    b5.to_csv(b5_path, index=False)
    print(f"-> Saved NASA B0005 cell profile ({len(b5)} cycles) to: {b5_path}")

    return final_df

if __name__ == "__main__":
    process_nasa_dataset()
