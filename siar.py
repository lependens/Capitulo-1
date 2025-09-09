import requests
import pandas as pd
import json
from datetime import datetime
import time
import os

# Configuración
API_KEY = '-bbkRtYLf7hUBgqUJ0J_BfVi_DD_2ATP-F5h_MR8-haK1udnC9'  # Tu clave de 50 caracteres
BASE_URL_DATOS = 'https://servicio.mapama.gob.es/apisiar/API/v1/datos'
BASE_URL_INFO = 'https://servicio.mapama.gob.es/apisiar/API/v1/Info'
TIPO_DATOS = 'Diarios'
AMBITO = 'Estacion'
OUTPUT_FILE = 'datos_baleares_manual.xlsx'

def consultar_datos(tipo_datos, ambito, id_ambito, fecha_inicio, fecha_fin):
    params = {
        'ClaveAPI': API_KEY,
        'Id': id_ambito,
        'FechaInicial': fecha_inicio,
        'FechaFinal': fecha_fin
    }
    url = f"{BASE_URL_DATOS}/{tipo_datos}/{ambito}"
    response = requests.get(url, params=params)
    print(f"DEBUG: Status code para {id_ambito} ({fecha_inicio} a {fecha_fin}): {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"DEBUG: MensajeRespuesta: {data.get('MensajeRespuesta')}")
        if data['MensajeRespuesta'] is None and data['Datos']:
            return data['Datos']
        else:
            print(f"Error o vacío en datos ({id_ambito}): {data.get('MensajeRespuesta', 'Datos vacíos')}")
            return None
    elif response.status_code == 403:
        print(f"Error 403 ({id_ambito}): Límite excedido. Pausando 5 min...")
        time.sleep(300)
        return None
    else:
        print(f"Error HTTP {response.status_code} ({id_ambito}): {response.text}")
        return None

def consultar_accesos():
    url = f"{BASE_URL_INFO}/Accesos?ClaveAPI={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['MensajeRespuesta'] is None:
            return data['Datos'][0]
    return None

# Estaciones de Baleares (tus códigos)
estaciones_ids = ['IB09'], # 'IB10', 'IB101', 'IB11', 'IB08', 'IB05', 'IB02', 'IB03', 'IB01', 'IB06', 'IB07', 'IB04']

# Test: Solo 2020, y solo primera estación IB01 para verificar cobertura
años = [2017]
test_estacion = ['IB01']  # Cambia a ['IB09'] si quieres probarla, pero usa IB01 para datos conocidos

# Cargar existentes
todos_los_datos = []
if os.path.exists(OUTPUT_FILE):
    try:
        df_existing = pd.read_excel(OUTPUT_FILE, sheet_name='Total')
        todos_los_datos = df_existing.to_dict('records')
        print(f"Cargados {len(todos_los_datos)} datos existentes.")
    except Exception as e:
        print(f"Error cargando Excel: {e}. Borrando archivo y empezando nuevo.")
        os.remove(OUTPUT_FILE) if os.path.exists(OUTPUT_FILE) else None

# Límites
accesos = consultar_accesos()
if accesos:
    print(f"Límites: {accesos.get('MaxAccesosDia')} peticiones/día, {accesos.get('MaxRegistrosDia')} registros/día")
    print(f"Usados: {accesos.get('NumAccesosDiaActual')} peticiones, {accesos.get('RegistrosAcumuladosDia')} registros")

# Extracción por estación, año y MES
peticiones_count = 0
for estacion_id in test_estacion:  # Usa test_estacion para limitar
    print(f"\n--- Estación {estacion_id} ---")
    for año in años:
        meses = pd.date_range(start=f"{año}-01-01", end=f"{año}-12-31", freq='MS')
        for fecha_mes in meses:
            fecha_inicio = fecha_mes.strftime('%Y-%m-%d')
            fin_mes = (fecha_mes + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
            print(f"  Mes {fecha_inicio} a {fin_mes}...")
            
            datos_mes = consultar_datos(TIPO_DATOS, AMBITO, estacion_id, fecha_inicio, fin_mes)
            if datos_mes:
                for d in datos_mes:
                    d['Estacion'] = estacion_id
                todos_los_datos.extend(datos_mes)
                print(f"    +{len(datos_mes)} registros.")
            else:
                print(f"    Sin datos para este mes.")
            
            peticiones_count += 1
            if peticiones_count % 3 == 0:  # Cada 3 para test
                accesos = consultar_accesos()
                if accesos:
                    print(f"    Usados ahora: {accesos.get('NumAccesosDiaActual')} peticiones, {accesos.get('RegistrosAcumuladosDia')} registros")
            time.sleep(90)  # Pausa 90s

# Guardar (CORREGIDO: Manejo correcto de mode e if_sheet_exists)
if todos_los_datos:
    print(f"\nTotal: {len(todos_los_datos)} registros")
    df = pd.DataFrame(todos_los_datos)
    if 'Fecha' in df.columns:
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        df = df.sort_values(['Estacion', 'Fecha'])
    
    # Manejo corregido del ExcelWriter
    mode = 'a' if os.path.exists(OUTPUT_FILE) else 'w'
    writer_kwargs = {'engine': 'openpyxl'}
    if mode == 'a':
        writer_kwargs['if_sheet_exists'] = 'replace'
    
    with pd.ExcelWriter(OUTPUT_FILE, mode=mode, **writer_kwargs) as writer:
        # Hoja total (si append, replace; si write, nueva)
        df.to_excel(writer, sheet_name='Total', index=False)
        # Hojas por estación-año para test
        for estacion_id in test_estacion:
            est_df = df[df['Estacion'] == estacion_id]
            for año in años:
                mask = (est_df['Fecha'].dt.year == año)
                datos_año_df = est_df[mask]
                if not datos_año_df.empty:
                    sheet_name = f"{estacion_id}_{año}"[:31]
                    datos_año_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"¡Excel actualizado! '{OUTPUT_FILE}'.")
    print("Vista previa:")
    print(df.head())
else:
    print("Sin datos nuevos. Verifica fechas (usa pasadas) o contacta MAPA.")