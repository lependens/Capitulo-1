import pandas as pd
import glob
import os

# Carpeta con CSVs de datos
CARPETA_DATOS = 'C:/Users/josep/OneDrive/Documentos/GitHub/Capitulo-1/datos_siar_baleares'

# Lista de columnas para medias históricas
columnas_stats = ['TempMedia', 'HumedadMedia', 'VelViento', 'Radiacion', 'Precipitacion', 'EtPMon']

# Cargar estaciones
df_estaciones = pd.read_csv(os.path.join(CARPETA_DATOS, 'estaciones_baleares.csv'))  # Asume que tienes este archivo

# Inicializar DataFrame para stats
df_stats = pd.DataFrame(columns=['Estacion', 'Nombre', 'Lat', 'Long'] + [f'Media_{col}' for col in columnas_stats])

for idx, row in df_estaciones.iterrows():
    codigo = row['Codigo']  # Asume columna 'Codigo'
    archivo = os.path.join(CARPETA_DATOS, f"{codigo}_datos_completos.csv")
    
    if os.path.exists(archivo):
        df_data = pd.read_csv(archivo)
        stats = df_data[columnas_stats].mean().to_dict()
        stats = {f'Media_{k}': v for k, v in stats.items()}
        
        nueva_fila = pd.DataFrame({
            'Estacion': [codigo],
            'Nombre': [row.get('Estacion', 'N/A')],  # Asume columna 'Estacion' o 'Nombre'
            'Lat': [row['Latitud']],  # Asume 'Latitud'
            'Long': [row['Longitud']],  # Asume 'Longitud'
            **stats
        })
        df_stats = pd.concat([df_stats, nueva_fila], ignore_index=True)
    else:
        print(f"Archivo {archivo} no encontrado.")

# Guardar stats
df_stats.to_csv(os.path.join(CARPETA_DATOS, 'estadisticas_historicas.csv'), index=False)
print("estadisticas_historicas.csv generado.")