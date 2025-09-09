import requests
import pandas as pd

# Configura tus credenciales y parámetros
usuario = '43227070L'
clave = '-bbkRtYLf7hUBgqUJ0J_BfVi_DD_2ATP-F5h_MR8-haK1udnC9'
estacion_id = 'ID_ESTACION'  # Cambia por el ID de tu estación
fecha_inicio = '2000-01-01'
fecha_fin = '2024-12-31'

# URL base de la API SIAR (modifica si es diferente)
url = f'https://www.siar.es/api/historico'

# Parámetros de la consulta
params = {
    'usuario': usuario,
    'clave': clave,
    'estacion': estacion_id,
    'fecha_inicio': fecha_inicio,
    'fecha_fin': fecha_fin,
    'formato': 'json'
}

# Realiza la petición
response = requests.get(url, params=params)
response.raise_for_status()  # Lanza error si la petición falla

# Procesa los datos
datos = response.json()
df = pd.DataFrame(datos)

# Guarda a CSV
df.to_csv('datos_siar.csv', index=False)
print('Datos guardados en datos_siar.csv')
