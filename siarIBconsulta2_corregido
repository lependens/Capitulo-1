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
                # Verificar si 'Fecha' está presente, si no, asignarla basada en el rango
                if 'Fecha' not in dato:
                    print(f"  - Advertencia: 'Fecha' no encontrada en un registro. Asignando {fecha_inicio}T00:00:00.")
                    dato['Fecha'] = fecha_inicio + "T00:00:00"  # Formato esperado por la API
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
fecha_inicio_input = input("Introduce la fecha inicial (YYYY-MM-DD): ").strip()
fecha_fin_input = input("Introduce la fecha final (YYYY-MM-DD): ").strip()

# Parsear fechas de input con el nuevo formato
try:
    fecha_inicio = datetime.strptime(fecha_inicio_input, '%Y-%m-%d')
    fecha_fin = datetime.strptime(fecha_fin_input, '%Y-%m-%d')
except ValueError:
    print("Formato de fecha inválido. Usa YYYY-MM-DD (ej. 2024-12-31).")
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
    # Verificar si 'Fecha' existe y convertirla
    if 'Fecha' not in df_existente.columns:
        print("Advertencia: La columna 'Fecha' no existe en los datos existentes. Intentando reconstruirla...")
        fechas_faltantes = pd.date_range(start='2000-01-01', periods=len(df_existente), freq='D')
        df_existente['Fecha'] = fechas_faltantes
    else:
        df_existente['Fecha'] = pd.to_datetime(df_existente['Fecha'], format='%Y-%m-%d', errors='coerce')
    fechas_existentes = set(df_existente['Fecha'].dt.date.dropna())  # Ignorar NaT
    print(f"Archivo existente encontrado con {len(df_existente)} registros.")
    print(f"Tipo de 'Fecha' en existente: {df_existente['Fecha'].dtype}")
else:
    df_existente = pd.DataFrame(columns=['Fecha', 'TempMedia', 'TempMax', 'HorMinTempMax', 'TempMin', 'HorMinTempMin',
                                        'HumedadMedia', 'HumedadMax', 'HorMinHumMax', 'HumedadMin', 'HorMinHumMin',
                                        'VelViento', 'DirViento', 'VelVientoMax', 'HorMinVelMax', 'DirVientoVelMax',
                                        'Radiacion', 'Precipitacion', 'TempSuelo1', 'TempSuelo2', 'EtPMon', 'PePMon', 'Estacion'])
    fechas_existentes = set()
    print("Archivo nuevo creado.")

# Generar rangos mensuales dentro del intervalo solicitado
datos_nuevos = []
fecha_actual = fecha_inicio
while fecha_actual <= fecha_fin:
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
    print("Columnas en df_nuevos:", df_nuevos.columns.tolist())  # Depuración
    if 'Fecha' not in df_nuevos.columns:
        print("Error: 'Fecha' no encontrada en datos nuevos. Revisar estructura de la API.")
    else:
        df_nuevos['Fecha'] = pd.to_datetime(df_nuevos['Fecha'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')
        # Convertir a formato sin hora para consistencia
        df_nuevos['Fecha'] = df_nuevos['Fecha'].dt.strftime('%Y-%m-%d')
    print(f"Tipo de 'Fecha' en nuevos: {df_nuevos['Fecha'].dtype}")
    
    # Convertir 'Fecha' en existente a string para consistencia
    if not df_existente.empty:
        df_existente['Fecha'] = df_existente['Fecha'].dt.strftime('%Y-%m-%d')
    
    df_combined = pd.concat([df_existente, df_nuevos], ignore_index=True)
    
    if 'Fecha' not in df_combined.columns:
        print("Error: La columna 'Fecha' no se combinó correctamente. Revisar datos.")
    else:
        df_combined['Fecha'] = pd.to_datetime(df_combined['Fecha'], errors='coerce')
        df_combined['Fecha'] = df_combined['Fecha'].dt.strftime('%Y-%m-%d')
        print(f"Tipo de 'Fecha' en combined antes de sort: {df_combined['Fecha'].dtype}")
    
    df_combined = df_combined.sort_values(by='Fecha', na_position='last')
    
    # Guardar el archivo actualizado con todas las columnas
    df_combined.to_csv(ARCHIVO_FINAL, index=False)
    print(f"\nPrimeras 5 filas del CSV resultante:\n{df_combined.head().to_string()}")
    print(f"\nÚltimas 5 filas del CSV resultante:\n{df_combined.tail().to_string()}")
    print(f"\nAgregados {len(datos_nuevos)} nuevos registros. Total ahora: {len(df_combined)} en {ARCHIVO_FINAL}")
else:
    print("\nNo se obtuvieron nuevos datos para agregar.")

print("\nProceso completado.")