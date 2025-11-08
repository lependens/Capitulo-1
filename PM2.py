import pandas as pd
import numpy as np
import math
import os
from datetime import datetime
import warnings
import logging  # Para logs futuros

# Path to data
data_path = r'C:/Users/josep/OneDrive/Documentos/GitHub/Capitulo-1/datos_siar_baleares'

# Config logging after data_path
logging.basicConfig(level=logging.INFO, filename=os.path.join(data_path, 'pm2_log.txt'), filemode='w')  # Log file para diagnósticos

# Function to convert DMS string to decimal degrees
def dms_to_dd(dms_str):
    if pd.isna(dms_str):
        raise ValueError("NaN in dms_str")
    dms_str = str(dms_str).strip()
    if dms_str.replace('.', '', 1).isdigit() or '.' in dms_str:
        return float(dms_str)
    if len(dms_str) < 7:
        raise ValueError(f"Too short dms_str: {dms_str}")
    direction = dms_str[-1].upper()
    if direction not in ['N', 'S', 'E', 'W']:
        raise ValueError(f"Invalid direction in dms_str: {dms_str}")
    dms = dms_str[:-1]
    dd = int(dms[0:2])
    mm = int(dms[2:4])
    ss = int(dms[4:6])
    ms = int(dms[6:]) if len(dms) > 6 else 0
    seconds = ss + ms / 1000.0
    decimal = dd + mm / 60.0 + seconds / 3600.0
    if direction in ['S', 'W']:
        decimal = -decimal
    return decimal

# Function to calculate saturation vapor pressure ea(T)
def saturation_vapor_pressure(T):
    return 0.611 * math.exp(17.27 * T / (T + 237.3))

# Function to calculate slope of saturation vapor pressure curve Delta
def slope_vapor_pressure(T, es):
    return 4098 * es / (T + 237.3)**2

# Function to calculate psychrometric constant gamma
def psychrometric_constant(P, lambda_):
    return 0.00163 * P / lambda_

# Function to calculate atmospheric pressure P
def atmospheric_pressure(z):
    return 101.3 * ((293 - 0.0065 * z) / 293)**5.26

# Function to calculate latent heat of vaporization lambda
def latent_heat_vaporization(T):
    return 2.501 - 0.002361 * T

# Function to calculate extraterrestrial radiation Ra
def extraterrestrial_radiation(lat, J):
    Gsc = 0.0820
    dr = 1 + 0.033 * math.cos(2 * math.pi * J / 365)
    delta = 0.409 * math.sin(2 * math.pi * J / 365 - 1.39)
    lat_rad = math.radians(lat)
    arg = -math.tan(lat_rad) * math.tan(delta)
    arg = max(min(arg, 1), -1)
    ws = math.acos(arg)
    Ra = (24 * 60 / math.pi) * Gsc * dr * (ws * math.sin(lat_rad) * math.sin(delta) + math.cos(lat_rad) * math.cos(delta) * math.sin(ws))
    return Ra

# Function to calculate daylight hours N
def daylight_hours(lat, J):
    delta = 0.409 * math.sin(2 * math.pi * J / 365 - 1.39)
    lat_rad = math.radians(lat)
    arg = -math.tan(lat_rad) * math.tan(delta)
    arg = max(min(arg, 1), -1)
    ws = math.acos(arg)
    N = 24 / math.pi * ws
    return N

# Function to calculate clear sky radiation Rso
def clear_sky_radiation(Ra, z):
    return Ra * (0.75 + 2e-5 * z)

# Function to calculate net longwave radiation Rnl
def net_longwave_radiation(Tmax, Tmin, ed, Rs, Rso):
    sigma = 4.903e-9
    TmaxK = Tmax + 273.15
    TminK = Tmin + 273.15
    if ed <= 0:
        ed = 0.0001
    emissivity = 0.34 - 0.14 * math.sqrt(ed)
    if Rs <= 0 or Rso <= 0:
        cloud_factor = 1.0
    else:
        cloud_factor = 1.35 * (Rs / Rso) - 0.35
        cloud_factor = max(0.2, min(1.0, cloud_factor))
    Rnl = sigma * ((TmaxK**4 + TminK**4) / 2) * emissivity * cloud_factor
    return Rnl

# Function to calculate ET0 Penman-Monteith
def penman_monteith_ET0(Delta, Rn, G, gamma, T_mean, U2, es, ed):
    num = 0.408 * Delta * (Rn - G) + gamma * (900 / (T_mean + 273)) * U2 * (es - ed)
    den = Delta + gamma * (1 + 0.34 * U2)
    if den == 0:
        logging.warning("Denominador = 0, retornando 0 para ET0")
        return 0
    return num / den

# Read stations data with normalization
estaciones_df = pd.read_csv(os.path.join(data_path, 'estaciones_baleares.csv'), sep=',', encoding='utf-8-sig', quotechar='"', engine='python')
print("Columnas detectadas en catálogo:", estaciones_df.columns.tolist())  # Debug
estaciones_df['Codigo'] = estaciones_df['Codigo'].astype(str).str.strip().str.upper()

# Main processing function
def process_station_data(code, estaciones_df):
    code = code.upper().strip()
    station_file = os.path.join(data_path, f'{code}_datos_depurados.csv')
    if not os.path.exists(station_file):
        print(f"File {station_file} not found. Verifica nombre/existencia.")
        return None
    
    matching_stations = estaciones_df[estaciones_df['Codigo'] == code]
    if matching_stations.empty:
        print(f"No station found with code {code}. Available codes: {', '.join(sorted(estaciones_df['Codigo'].unique()))}")
        return None
    
    station_info = matching_stations.iloc[0]
    lat = dms_to_dd(station_info['Latitud'])
    z = station_info['Altitud']
    
    df = pd.read_csv(station_file, sep=',', encoding='utf-8-sig')
    print(f"Cargados {len(df)} filas de {station_file}. Columnas: {df.columns.tolist()}")  # Debug: check tamaño y columnas
    print("Primeras 5 filas cargadas:\n", df.head(5).to_string())  # Debug extra: ver datos reales cargados
    
    # Drop filas con NaN en columnas requeridas
    required_cols = ['TempMax', 'TempMin', 'TempMedia', 'VelViento', 'Radiacion', 'HumedadMax', 'HumedadMin', 'EtPMon', 'Fecha']
    df.dropna(subset=required_cols, inplace=True)
    print(f"Filas después de drop NaN: {len(df)}")  # Debug: ver si drop eliminó todo
    
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['J'] = df['Fecha'].dt.dayofyear
    df['N'] = np.nan
    df['ET0_calc'] = np.nan
    df['diff'] = np.nan
    df['ET0_sun'] = np.nan
    df['diff_sun'] = np.nan
    df['Ra'] = np.nan
    
    processed_rows = 0
    for idx, row in df.iterrows():
        try:
            Tmax = row['TempMax']
            Tmin = row['TempMin']
            T_mean = row['TempMedia']
            U2 = row['VelViento']
            Rs = row['Radiacion']
            RHmax = row['HumedadMax']
            RHmin = row['HumedadMin']
            
            ea_max = saturation_vapor_pressure(Tmax)
            ea_min = saturation_vapor_pressure(Tmin)
            es = (ea_max + ea_min) / 2
            ed = (ea_max * (RHmin / 100) + ea_min * (RHmax / 100)) / 2
            
            lambda_ = latent_heat_vaporization(T_mean)
            P = atmospheric_pressure(z)
            gamma = psychrometric_constant(P, lambda_)
            Delta = slope_vapor_pressure(T_mean, saturation_vapor_pressure(T_mean))
            J = row['J']
            Ra = extraterrestrial_radiation(lat, J)
            df.at[idx, 'Ra'] = Ra
            Rso = clear_sky_radiation(Ra, z)
            N = daylight_hours(lat, J)
            df.at[idx, 'N'] = N
            
            Rns = (1 - 0.23) * Rs
            Rnl = net_longwave_radiation(Tmax, Tmin, ed, Rs, Rso)
            Rn = Rns - Rnl
            G = 0
            ET0_calc = penman_monteith_ET0(Delta, Rn, G, gamma, T_mean, U2, es, ed)
            df.at[idx, 'ET0_calc'] = ET0_calc
            df.at[idx, 'diff'] = row['EtPMon'] - ET0_calc
            
            Rs_sun = Rso
            Rnl_sun = net_longwave_radiation(Tmax, Tmin, ed, Rs_sun, Rso)
            Rn_sun = (1 - 0.23) * Rs_sun - Rnl_sun
            ET0_sun = penman_monteith_ET0(Delta, Rn_sun, G, gamma, T_mean, U2, es, ed)
            df.at[idx, 'ET0_sun'] = ET0_sun
            df.at[idx, 'diff_sun'] = row['EtPMon'] - ET0_sun
            
            print(f"Fila {idx}: ET0_calc = {ET0_calc:.4f}, diff = {df.at[idx, 'diff']:.4f}")  # Debug por fila
            processed_rows += 1
        except Exception as e:
            print(f"Error en fila {idx}: {e}. Saltando fila.")
            logging.error(f"Error en fila {idx} de {code}: {e}")
            df.at[idx, 'ET0_calc'] = np.nan  # Mark as NaN
    
    output_file = os.path.join(data_path, f'{code}_et0_calc.csv')
    df.to_csv(output_file, index=False)
    print(f"Processed {code}, saved to {output_file}. Filas procesadas: {processed_rows}/{len(df)}")
    if processed_rows > 0:
        print(f"Mean diff = {df['diff'].mean():.4f} mm/day")  # Métrica final si hay datos
    else:
        print("No filas procesadas – check NaN o columnas.")
    return df

# Interactive
while True:
    code = input("Enter station code (e.g., IB01) or 'quit' to exit: ").strip()
    if code.lower() == 'quit':
        break
    process_station_data(code, estaciones_df)
    print("\n")

print("Processing complete.")