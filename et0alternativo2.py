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
    
    # Hargreaves-Samani estándar con Ra (hg2 from TFG, including 0.408 for unit consistency as in MATLAB code)
    TD = df['TempMax'] - df['TempMin']
    T = df['TempMedia']
    Ra = df['Ra']
    ET0_harg_ra = 0.0023 * 0.408 * Ra * (T + 17.8) * TD**0.5
    df['ET0_harg_ra'] = ET0_harg_ra
    df['diff_harg_ra'] = df['EtPMon'] - ET0_harg_ra
    
    # Nueva columna: Hargreaves original con Rs medida (basado en fórmula proporcionada: 0.0135 * Rs * (T + 17.8), con 0.408 para conversión de unidades MJ/m²/d a mm/d)
    Rs = df['Radiacion']  # Columna 'Radiacion' en MJ/m²/d
    ET0_harg_rs = 0.0135 * 0.408 * Rs * (T + 17.8)
    df['ET0_harg_rs'] = ET0_harg_rs
    df['diff_harg_rs'] = df['EtPMon'] - ET0_harg_rs
    
    # Valiantzas variant (hg3 from TFG, assuming typo in '1.001' as '1 -' for (1 - RH/100)^0.2; using HumedadMedia as RH)
    RH = df['HumedadMedia']  # Assuming col13 is HumedadMedia; change to 'HumedadMin' if needed
    ET0_val = 0.408 * 0.0135 * 0.338 * Ra * (TD ** 0.3) * ((1 - RH / 100) ** 0.2) * (T + 17.8)
    df['ET0_val'] = ET0_val
    df['diff_val'] = df['EtPMon'] - ET0_val
    
    output_file = os.path.join(data_path, f'{code}_et0_variants.csv')
    df.to_csv(output_file, index=False)
    print(f"Processed variants for {code}, saved to {output_file}")
    print(f"Mean diff_harg_ra = {df['diff_harg_ra'].mean():.4f} mm/day")
    print(f"Mean diff_harg_rs = {df['diff_harg_rs'].mean():.4f} mm/day")
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