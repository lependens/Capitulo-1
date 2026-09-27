import pandas as pd
import requests
import json
import time
from datetime import datetime, timedelta
import os

# Configuración
API_BASE = "https://servicio.mapama.gob.es/apisiar/API/v1"
TIPO_DATOS = "Diarios"  # Datos diarios
AMBITO = "Estacion"
CLAVE_API = "-bbkRtYLf7hUBgqUJ0J_BfVi_DD_2ATP-F5h_MR8-haK1udnC9"  # Tu clave API
CARPETA_SALIDA = 'C:/Users/josep/OneDrive/Documentos/GitHub/Capitulo-1/datos_siar_baleares'

# Función para obtener datos de un rango de fechas
def fetch_data_rango(codigo, fecha_inicio, fecha_fin, clave_api):
    url = f"{API_BASE}/Datos/{TIPO_DATOS}/{AMBITO}?Id={codigo}&FechaInicial={fecha_inicio}&FechaFinal={fecha_fin}&ClaveAPI={clave_api}"
    print(f"Intentando: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data_json = response.json()
        if "Datos" in data_json and data_json["Datos"]:
            for dato in data_json["Datos"]:
                dato["Estacion"] = codigo
            print(f"  - Descargados {len(data_json['Datos'])} registros para {fecha_inicio} a {fecha_fin}")
            return data_json["Datos"]
        else:
            print(f"  - No hay datos para {fecha_inicio} a {fecha_fin}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"  - Error: {e}")
        if 'response' in locals() and response.text:
            print(f"    Respuesta API: {response.text}")
        return []

# Obtener inputs del usuario
codigo_estacion = input("Introduce el ID de la estación (ej. IB01): ").strip()
fecha_inicio_input = input("Introduce la fecha inicial (dd/mm/yyyy): ").strip()
fecha_fin_input = input("Introduce la fecha final (dd/mm/yyyy): ").strip()

# Parsear fechas de input
try:
    fecha_inicio = datetime.strptime(fecha_inicio_input, '%d/%m/%Y')
    fecha_fin = datetime.strptime(fecha_fin_input, '%d/%m/%Y')
except ValueError:
    print("Formato de fecha inválido. Usa dd/mm/yyyy.")
    exit()

fecha_inicio_str = fecha_inicio.strftime('%Y-%m-%d')
fecha_fin_str = fecha_fin.strftime('%Y-%m-%d')

# Archivo final
ARCHIVO_FINAL = os.path.join(CARPETA_SALIDA, f"{codigo_estacion}_datos_completos.csv")

# Crear carpeta si no existe
os.makedirs(CARPETA_SALIDA, exist_ok=True)

# Cargar datos existentes si el archivo existe
if os.path.exists(ARCHIVO_FINAL):
    df_existente = pd.read_csv(ARCHIVO_FINAL)
    # Convertir 'Fecha' a datetime con manejo de errores
    df_existente['Fecha'] = pd.to_datetime(df_existente['Fecha'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')
    fechas_existentes = set(df_existente['Fecha'].dt.date.dropna())  # Ignorar NaT
    print(f"Archivo existente encontrado con {len(df_existente)} registros.")
    print(f"Tipo de 'Fecha' en existente: {df_existente['Fecha'].dtype}")
else:
    df_existente = pd.DataFrame()
    fechas_existentes = set()
    print("Archivo nuevo creado.")

# Generar rangos mensuales dentro del intervalo solicitado
datos_nuevos = []
fecha_actual = fecha_inicio
while fecha_actual <= fecha_fin:
    # Calcular fin del mes
    primer_dia_siguiente_mes = (fecha_actual.replace(day=28) + timedelta(days=4)).replace(day=1)
    fecha_fin_mes = primer_dia_siguiente_mes - timedelta(days=1)
    if fecha_fin_mes > fecha_fin:
        fecha_fin_mes = fecha_fin
    
    fecha_inicio_str_mes = fecha_actual.strftime('%Y-%m-%d')
    fecha_fin_str_mes = fecha_fin_mes.strftime('%Y-%m-%d')
    
    # Fetch data
    datos_mes = fetch_data_rango(codigo_estacion, fecha_inicio_str_mes, fecha_fin_str_mes, CLAVE_API)
    
    # Filtrar para evitar duplicados
    datos_nuevos_mes = [dato for dato in datos_mes if pd.to_datetime(dato['Fecha']).date() not in fechas_existentes]
    datos_nuevos.extend(datos_nuevos_mes)
    
    # Pausa de 60 segundos para respetar límite
    time.sleep(60)
    
    # Avanzar al siguiente mes
    fecha_actual = primer_dia_siguiente_mes

# Agregar nuevos datos al existente
if datos_nuevos:
    df_nuevos = pd.json_normalize(datos_nuevos)
    # Convertir 'Fecha' en nuevos a datetime
    df_nuevos['Fecha'] = pd.to_datetime(df_nuevos['Fecha'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')
    print(f"Tipo de 'Fecha' en nuevos: {df_nuevos['Fecha'].dtype}")
    
    df_combined = pd.concat([df_existente, df_nuevos], ignore_index=True)
    
    # Convertir 'Fecha' en combined a datetime (por si acaso)
    df_combined['Fecha'] = pd.to_datetime(df_combined['Fecha'], errors='coerce')
    print(f"Tipo de 'Fecha' en combined antes de sort: {df_combined['Fecha'].dtype}")
    
    # Ordenar por fecha (NaT al final)
    df_combined = df_combined.sort_values(by='Fecha', na_position='last')
    
    # Guardar el archivo actualizado
    df_combined.to_csv(ARCHIVO_FINAL, index=False)
    print(f"\nAgregados {len(datos_nuevos)} nuevos registros. Total ahora: {len(df_combined)} en {ARCHIVO_FINAL}")
else:
    print("\nNo se obtuvieron nuevos datos para agregar.")

print("\nProceso completado.")