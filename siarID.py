import requests
import pandas as pd
import json

# Configuración
API_KEY = '-bbkRtYLf7hUBgqUJ0J_BfVi_DD_2ATP-F5h_MR8-haK1udnC9'  # Reemplaza con tu clave de 50 caracteres
BASE_URL_INFO = 'https://servicio.mapama.gob.es/apisiar/API/v1/Info'
OUTPUT_LISTADO = 'estaciones_listado.xlsx'  # Archivo Excel de salida

def consultar_estaciones():
    """Consulta todas las estaciones autorizadas."""
    url = f"{BASE_URL_INFO}/Estaciones?ClaveAPI={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['MensajeRespuesta'] is None:
            return data['Datos']
        else:
            print(f"Error: {data['MensajeRespuesta']}")
            return None
    else:
        print(f"Error HTTP {response.status_code}: {response.text}")
        return None

# Obtener y guardar listado
print("Obteniendo listado de todas las estaciones autorizadas...")
estaciones = consultar_estaciones()
if estaciones:
    print(f"Total estaciones: {len(estaciones)}")
    
    # Convertir a DataFrame
    df = pd.DataFrame(estaciones)
    
    # Guardar en Excel
    df.to_excel(OUTPUT_LISTADO, index=False)
    print(f"Listado guardado en '{OUTPUT_LISTADO}'. Abre en Excel y filtra por 'Baleares' o códigos 'IBxx'.")
    
    # Vista previa en consola (primeras 10)
    print("\nVista previa (primeras 10):")
    print(df[['Codigo', 'Estacion', 'Termino', 'Latitud', 'Longitud']].head(10))
    
    # Buscar posibles Baleares en consola (para preview)
    print("\nPosibles estaciones Baleares (buscando 'BALEAR' o 'IB'): ")
    baleares_posibles = df[df['Estacion'].str.contains('BALEAR|IBIZA|MALLORCA|MALLORCA|PALMA|MAHON|ILLES', na=False) | df['Termino'].str.contains('BALEAR|IBIZA|MALLORCA|MALLORCA|PALMA|MAHON|ILLES', na=False) | df['Codigo'].str.contains('IB', na=False)]
    for _, row in baleares_posibles.iterrows():
        print(f"- Codigo: {row['Codigo']}, Estacion: {row['Estacion']}, Termino: {row['Termino']}")
else:
    print("No se obtuvieron estaciones. Verifica tu clave API.")