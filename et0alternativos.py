import pandas as pd
import numpy as np
import math
import os
from datetime import datetime

# Path to data
data_path = r'C:/Users/josep/OneDrive/Documentos/GitHub/Capitulo-1/datos_siar_baleares'

# Main processing function
def process_station_variants(code):
    station_file = os.path.join(data_path, f'{code}_et0_calc.csv')
    if not os.path.exists(station_file):
        print(f"File {station_file} not found. Run PM2.py first.")
        return None
    
    df = pd.read_csv(station_file, sep=',')
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Hargreaves / Hargreaves-Samani (equations 7 and 8 are the same)
    TD = df['TempMax'] - df['TempMin']
    T = df['TempMedia']
    Ra = df['Ra']
    ET0_harg = 0.0023 * Ra * (T + 17.8) * TD**0.5
    df['ET0_harg'] = ET0_harg
    df['diff_harg'] = df['EtPMon'] - ET0_harg
    
    # Valiantzas (assuming equation 16 is a typo and uses Rs instead of Ra)
    # Rs = Ra * (a + b * RHmin / 100)
    # Here, using example values for a and b (empirical coefficients); adjust based on your TFG calibration for Baleares
    a = 0.75  # Example value; replace with actual from TFG
    b = -0.5  # Example value (negative to account for higher humidity reducing Rs); replace with actual
    RHmin = df['HumedadMin']
    Rs_est = Ra * (a + b * (RHmin / 100))
    ET0_val = 0.0023 * Rs_est * (T + 17.8) * TD**0.5
    df['Rs_est_val'] = Rs_est  # Optional: save estimated Rs
    df['ET0_val'] = ET0_val
    df['diff_val'] = df['EtPMon'] - ET0_val
    
    output_file = os.path.join(data_path, f'{code}_et0_variants.csv')
    df.to_csv(output_file, index=False)
    print(f"Processed variants for {code}, saved to {output_file}")
    print(f"Mean diff_harg = {df['diff_harg'].mean():.4f} mm/day")
    print(f"Mean diff_val = {df['diff_val'].mean():.4f} mm/day")
    return df

# Interactive
while True:
    code = input("Enter station code (e.g., IB01) or 'quit' to exit: ").strip()
    if code.lower() == 'quit':
        break
    process_station_variants(code)
    print("\n")

print("Processing complete.")