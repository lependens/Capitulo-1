# 1.2 Depuración y verificación de datos meteorológicos

**Objetivo**

Depurar los datos diarios de las estaciones meteorológicas de las Islas Baleares y, posteriormente, verificar los valores de **ET₀** proporcionados por el SIAR mediante diferentes modelos de cálculo.

---

## 1.2.1 Depuración de datos

La calidad de los datos es fundamental para obtener resultados confiables.  
Por ello, aplicaremos las siguientes transformaciones:

1. **Eliminación de valores nulos (NaN).**  
   - Se eliminan todas las filas que contengan datos faltantes en las columnas de interés.

2. **Detección y filtrado de valores extremos.**  
   - Para cada variable meteorológica relevante, se calcula la **media** y la **desviación estándar (σ)**.  
   - Se eliminan todas las filas que superen **± 3σ** respecto a la media.  
   - Este procedimiento ayuda a reducir el impacto de datos anómalos o errores de medición.

---

## 1.2.2 Cálculo de ET₀ con Penman-Monteith

Una vez depurados los datos, el siguiente paso es verificar los valores de ET₀ proporcionados por el SIAR.  
Para ello, se implementará la fórmula de **FAO Penman-Monteith**, considerada el estándar:

$$
ET_0 = \frac{0.408 \, \Delta (R_n - G) + \gamma \, \frac{900}{T+273} \, u_2 (e_s - e_a)}{\Delta + \gamma (1+0.34 \, u_2)}
$$

Donde:  
- \(R_n\) = Radiación neta [MJ/m²/día]  
- \(G\) = Flujo de calor en el suelo [MJ/m²/día]  
- \(T\) = Temperatura media del aire [°C]  
- \(u_2\) = Velocidad del viento a 2 m [m/s]  
- \(e_s\) = Presión de vapor de saturación [kPa]  
- \(e_a\) = Presión de vapor real [kPa]  
- \(\Delta\) = Pendiente de la curva de presión de vapor [kPa/°C]  
- \(\gamma\) = Constante psicrométrica [kPa/°C]  

---

## 1.2.3 Estimaciones alternativas de ET₀

Además del modelo de Penman-Monteith, se evaluarán métodos simplificados y empíricos.

### 🔹 1.2.3.1 Ecuación original de Hargreaves (1985)

Basada únicamente en radiación solar y temperatura media:

$$
ET_0 = 0.0023 \, R_a \, (T + 17.8) \, (T_{max} - T_{min})^{0.5} \quad (7)
$$

Donde \(R_a\) es la radiación extraterrestre.

---

### 🔹 1.2.3.2 Ecuación de Hargreaves y Samani (1982, 1985)

Incluye la relación entre radiación extraterrestre y diferencia térmica:

$$
ET_0 = 0.0023 \, R_a \, (T + 17.8) \, (T_{max} - T_{min})^{0.5} \quad (8)
$$

El cálculo de \(R_a\) se obtiene de Allen et al. (1998):

$$
R_a = \frac{24 \times 60}{\pi} G_{sc} \left[ d_r \, \big( \omega_s \sin(\phi)\sin(\delta) + \cos(\phi)\cos(\delta)\sin(\omega_s) \big) \right] \quad (10)
$$

Donde:  
- \(G_{sc} = 0.082 \, MJ \, m^{-2} \, min^{-1}\) (constante solar).  
- \(d_r\) = distancia relativa Tierra-Sol:  

$$
d_r = 1 + 0.033 \cos\left(\frac{2 \pi J}{365}\right) \quad (11)
$$

- \(\delta\) = declinación solar:  

$$
\delta = 0.409 \sin\left(\frac{2 \pi J}{365} - 1.39\right) \quad (12)
$$

- \(\omega_s\) = ángulo horario al ocaso:  

$$
\omega_s = \arccos(-\tan(\phi)\tan(\delta)) \quad (13)
$$

Donde:  
- \(J\) = día juliano.  
- \(\phi\) = latitud en radianes.  

---

### 🔹 1.2.3.3 Ecuación de Valiantzas (2017)

Este modelo estima la radiación solar en función de la humedad relativa mínima:

$$
R_s = R_a \, \left[a + b \frac{RH_{min}}{100}\right] \quad (15)
$$

Donde \(a\) y \(b\) son coeficientes empíricos.  
Al combinar esta relación con la ecuación de Hargreaves, se obtiene:

$$
ET_0 = 0.0023 \, R_a \, (T + 17.8) \, (T_{max} - T_{min})^{0.5} \quad (16)
$$

---

## 1.2.4 Comparación de resultados

Se realizará una **comparación sistemática** entre:  
- ET₀ de SIAR.  
- ET₀ calculado con Penman-Monteith.  
- ET₀ estimado con Hargreaves, Hargreaves-Samani y Valiantzas.  

Esto permitirá validar la consistencia de los datos y analizar la aplicabilidad de métodos alternativos en las Islas Baleares.

---
## 21/09/2025: Prrimeras verisones de filtrado

## filtra.py

Recoge IBXX_datos_completos.csv y lo filtra analizando las columnas de variables.

Variables analizadas: El análisis de ±3σ se ejecuta en las columnas de columnas_interes = ['TempMedia', 'TempMax', 'TempMin', 'HumedadMedia', 'HumedadMax', 'HumedadMin', 'VelViento', 'DirViento', 'VelVientoMax', 'DirVientoVelMax', 'Radiacion', 'Precipitacion', 'EtPMon']. Estas son las variables numéricas relevantes para detectar outliers.

**Como funciona:**

- Pregunta por la estación: El script pregunta por el ID de la estación y genera las rutas de archivos basadas en él (ej. IB01_datos_completos.csv).

- Archivos nuevos: Crea IB01_datos_depurados.csv (datos filtrados) y IB01_datos_eliminados.csv (nulos + outliers).

- Gráficas: Para cada variable en columnas_interes, si hay valores eliminados, crea una gráfica scatter de los valores eliminados vs su índice, con una línea horizontal para la media (del original). Las guarda como PNG en graficas_eliminados (ej. IB01_TempMedia_eliminados.png).

## 21/09/2025: Cálculo de ET₀ con Penman-Monteith

**Objetivo:** Implementar el cálculo de la evapotranspiración de referencia (ET₀) utilizando la ecuación FAO Penman-Monteith para verificar los valores proporcionados por el SIAR en los datos depurados. Esto permite validar la consistencia de los datos y preparar el terreno para comparaciones con otros modelos empíricos. Se procesan los datos por estación, agregando columnas calculadas como ET₀ calculado, diferencias con SIAR y estimaciones alternativas bajo suposiciones de cielo claro.

**Pasos principales:**
1. Cargar los datos depurados de cada estación (IBXX_datos_completos.csv).
2. Obtener información geográfica (latitud, altitud) de estaciones_baleares.csv.
3. Calcular variables auxiliares: presión atmosférica, constante psicrométrica, radiación extraterrestre (Ra), radiación de cielo claro (Rso), radiación neta (Rn), etc.
4. Aplicar la fórmula Penman-Monteith para obtener ET₀ calculado.
5. Calcular diferencias con el ET₀ de SIAR (EtPMon) y una versión alternativa asumiendo cielo claro (ET0_sun).
6. Guardar resultados en un nuevo CSV (IBXX_et0_calc.csv) con métricas como la diferencia media.

La implementación se basa en la ecuación estándar de FAO Penman-Monteith (detallada en la sección 1.2.2), adaptada con funciones para manejar conversiones de coordenadas (DMS a decimal) y cálculos diarios.

---

## 1.3.1 Implementación en Python

Se ha desarrollado un script interactivo que procesa estaciones individuales, solicitando el código de estación (e.g., IB01) y generando el CSV con cálculos. A priori, lanza resultados correctos, con diferencias medias cercanas a cero en pruebas iniciales, lo que sugiere una buena alineación con los datos de SIAR.

### PM2.py

Este script realiza los cálculos principales. Incluye manejo de advertencias, conversión de fechas y exportación de resultados.

```python
import pandas as pd
import numpy as np
import math
import os
from datetime import datetime
import warnings

# Path to data
data_path = r'C:/Users/josep/OneDrive/Documentos/GitHub/Capitulo-1/datos_siar_baleares'

# Function to convert DMS string to decimal degrees
def dms_to_dd(dms_str):
    """
    Convert DMS string like '390036000N' to decimal degrees.
    Assumes format DDMMSSmmm with last letter N/S/E/W.
    If already decimal, return float.
    """
    if pd.isna(dms_str):
        raise ValueError("NaN in dms_str")
    dms_str = str(dms_str).strip()
    if dms_str.replace('.', '', 1).isdigit() or '.' in dms_str:  # It's decimal or scientific
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

# Read stations data
estaciones_df = pd.read_csv(os.path.join(data_path, 'estaciones_baleares.csv'), sep=',')

# Function to calculate saturation vapor pressure ea(T)
def saturation_vapor_pressure(T):
    return 0. Kung611 * math.exp(17.27 * T / (T + 237.3))

# Function to calculate slope of saturation vapor pressure (Delta)
def slope_vapor_pressure(T, es):
    return 4098 * es / (T + 237.3)**2

# Function to calculate psychrometric constant gamma
def psychrometric_constant(P,_lambda_):
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
    delta_rad = delta
    arg = -math.tan(lat_rad) * math.tan(delta_rad)
    arg = max(min(arg, 1), -1)
    Ws = math.acos(arg)
    Ra = (24 * 60 / math.pi) * Gsc * dr * (ws * math.sin(lat_rad) * math.sin(delta_rad) + math.cos(lat_rad) * math.cos(delta_rad) * math.sin(ws))
    return Ra

# Function to calculate daylight hours N
def daylight_hours(lat, J):
    delta = 0.409 * math.sin(2 * math.pi * J / 365 - 1.39)
    lat_rad = math.radians(lat)
    delta_rad = delta
    arg = -math.tan(lat_rad) * math.tan(delta_rad)
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
    if ed (<= 0:
        ed = 0.0001
    emissivity = 0.34 - 0.14 * math.sqrt(ed)
    if Rs <= 0 or Rso <= 0:
        cloud_factor = 1.0
    else:
        cloud_factor = 1.35 * (Rs / Rso) - 0.35
        cloud_factor = max(0.2, min(1.0, cloud_factor))
    Rnl = sigma * ((TmaxK**4 + TminK**) / 2) * emissivity * cloud_factor
    return Rnl

# Function to calculate ET0 Penman-Monteith
def penman_monteith_ET0(Delta, Rn, G, gamma, T_mean, U2, es, ed):
    num = 0.408 * Delta * (Rn - G) + gamma * (900 / (T_mean + 273)) * U2 * (es - ed)
    den = Delta + gamma * (1 + 0.34 * U2)
    if den == 0:
        return 0
    return num / den

# Main processing function
def process_station_data(code, estaciones_df):
    station_file = os.path.join(data_path, f'{code}_datos_completos.csv')
    if not os.path.exists(station_file):
        print(f"File {station_file} not found.")
        return None
    
    matching_stations = estaciones_df[estaciones_df['Codigo'] == code]
    if matching_stations.empty:
        print(f"No station found with code {code}.")
        return None
    
    station_info = matching_stations.iloc[0]
    lat = dms_to_dd(station_info['Latitud'])
    z = station_info['Altitud']
    
    df = pd.read_csv(station_file, sep=',')
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['J'] = df['Fecha'].dt.dayofyear
    df['N'] = np.nan  # Daylight hours
    df['ET0_calc'] = np.nan
    df['diff'] = np.nan
    df['ET0_sun'] = np.nan  # ET0 with sunshine (clear sky assumption)
    df['diff_sun'] = np.nan
    
    for idx, row in df.iterrows():
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
        Rso = clear_sky_radiation(Ra, z)
        N = daylight_hours(lat, J)
        df.at[idx, 'N'] = N
        
        # Standard ET0 with measured Rs
        Rns = (1 - 0.23) * Rs
        Rnl = net_longwave_radiation(Tmax, Tmin, ed, Rs, Rso)
        Rn = Rns - Rnl
        G = 0
        ET0_calc = penman_monteith_ET0(Delta, Rn, G, gamma, T_mean, U2, es, ed)
        df.at[idx, 'ET0_calc'] = ET0_calc
        df.at[idx, 'diff'] = row['EtPMon'] - ET0_calc
        
        # ET0 with sunshine hours (assuming clear sky, Rs_sun = Rso, equivalent to n/N=1)
        Rs_sun = Rso
        Rnl_sun = net_longwave_radiation(Tmax, Tmin, ed, Rs_sun, Rso)
        Rn_sun = (1 - 0.23) * Rs_sun - Rnl_sun
        ET0_sun = penman_monteith_ET0(Delta, Rn_sun, G, gamma, T_mean, U2, es, ed)
        df.at[idx, 'ET0_sun'] = ET0_sun
        df.at[idx, 'diff_sun'] = row['EtPMon'] - ET0_sun
    
    output_file = os.path.join(data_path, f'{code}_et0_calc.csv')
    df.to_csv(output_file, index=False)
    print(f"Processed {code}, saved to {output_file}")
    print(f"Mean diff = {df['diff'].mean():.4f} mm/day")
    print(f"Mean diff_sun = {df['diff_sun'].mean():.4f} mm/day")
    return df

# Interactive
while True:
    code = input("Enter station code (e.g., IB01) or 'quit' to exit: ").strip()
    if code.lower() == 'quit':
        break
    process_station_data(code, estaciones_df)
    print("\n")

print("Processing complete.")