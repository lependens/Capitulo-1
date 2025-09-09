import requests
import pandas as pd

# Configura tus credenciales
usuario = '43227070L'
clave = '-bbkRtYLf7hUBgqUJ0J_BfVi_DD_2ATP-F5h_MR8-haK1udnC9'

# 1. Obtener lista de estaciones
url_estaciones = 'https://www.siar.es/api/estaciones'
params_estaciones = {
    'usuario': usuario,
    'clave': clave,
    'formato': 'json'
}

resp = requests.get(url_estaciones, params=params_estaciones)
resp.raise_for_status()
estaciones = resp.json()

print("Tus estaciones disponibles:")
for est in estaciones:
    print(f"ID: {est['id']}, Nombre: {est['nombre']}")

# ...existing code...