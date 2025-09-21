import pandas as pd
import numpy as np
from scipy import stats  # Para calcular desviación estándar
import matplotlib.pyplot as plt  # Para crear gráficas
import os

# Ruta base para los archivos (ajusta si es necesario)
CARPETA_SALIDA = 'C:/Users/josep/OneDrive/Documentos/GitHub/Capitulo-1/datos_siar_baleares'
CARPETA_GRAFICAS = os.path.join(CARPETA_SALIDA, 'graficas_eliminados')
os.makedirs(CARPETA_GRAFICAS, exist_ok=True)  # Crear carpeta para gráficas si no existe

# Preguntar por la estación
codigo_estacion = input("Introduce el ID de la estación (ej. IB01): ").strip()

# Rutas de archivos basadas en la estación
archivo_entrada = os.path.join(CARPETA_SALIDA, f"{codigo_estacion}_datos_completos.csv")
archivo_depurado = os.path.join(CARPETA_SALIDA, f"{codigo_estacion}_datos_depurados.csv")  # Salida datos filtrados
archivo_eliminados = os.path.join(CARPETA_SALIDA, f"{codigo_estacion}_datos_eliminados.csv")  # Salida datos eliminados

# Verificar si el archivo de entrada existe
if not os.path.exists(archivo_entrada):
    print(f"Error: El archivo {archivo_entrada} no existe. Asegúrate de que los datos para {codigo_estacion} han sido descargados.")
    exit()

# Cargar el CSV
df_original = pd.read_csv(archivo_entrada)  # Guardar copia del original
df = df_original.copy()  # Trabajar con una copia para el filtrado

# Convertir 'Fecha' a datetime para manejo correcto
df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')

# 1. Eliminación de valores nulos (NaN) en columnas de interés
# Columnas relevantes según tu estructura (agrega o quita según necesites)
columnas_interes = ['TempMedia', 'TempMax', 'TempMin', 'HumedadMedia', 'HumedadMax', 'HumedadMin', 'VelViento', 'DirViento', 'VelVientoMax', 'DirVientoVelMax', 'Radiacion', 'Precipitacion', 'EtPMon']

# Eliminar filas con NaN en estas columnas y guardar las eliminadas
df_nulos_eliminados = df[df[columnas_interes].isna().any(axis=1)]
df = df.dropna(subset=columnas_interes)

print(f"Después de eliminar NaN: {len(df)} registros restantes.")

# 2. Detección y filtrado de valores extremos (±3σ)
# Guardar los eliminados por cada columna
df_eliminados_total = pd.DataFrame()  # DataFrame para acumular eliminados

for col in columnas_interes:
    media = df[col].mean()
    std = df[col].std()
    # Identificar filas a eliminar (fuera de ±3σ)
    df_a_eliminar = df[(df[col] < media - 3 * std) | (df[col] > media + 3 * std)]
    df_eliminados_total = pd.concat([df_eliminados_total, df_a_eliminar]).drop_duplicates()
    # Filtrar filas donde el valor está dentro de ±3σ
    df = df[(df[col] >= media - 3 * std) & (df[col] <= media + 3 * std)]
    print(f"Después de filtrar ±3σ en {col}: {len(df)} registros restantes.")

# Combinar eliminados (nulos + outliers)
df_eliminados_total = pd.concat([df_nulos_eliminados, df_eliminados_total]).drop_duplicates()

# Guardar el CSV depurado y los eliminados
df.to_csv(archivo_depurado, index=False)
df_eliminados_total.to_csv(archivo_eliminados, index=False)
print(f"\nDatos depurados guardados en {archivo_depurado}. Total final: {len(df)} registros.")
print(f"Datos eliminados guardados en {archivo_eliminados}. Total eliminados: {len(df_eliminados_total)} registros.")

# 3. Crear gráficas para cada variable eliminada comparada con la media
for col in columnas_interes:
    if col in df_eliminados_total.columns and not df_eliminados_total[col].empty:
        media = df_original[col].mean()  # Usar media del original para comparación
        valores_eliminados = df_eliminados_total[col].dropna()
        
        if not valores_eliminados.empty:
            fig, ax = plt.subplots()
            ax.scatter(valores_eliminados.index, valores_eliminados, color='red', label='Valores eliminados')
            ax.axhline(y=media, color='blue', linestyle='--', label=f'Media ({media:.2f})')
            ax.set_title(f"Valores eliminados vs Media en {col}")
            ax.set_xlabel("Índice de fila eliminada")
            ax.set_ylabel("Valor")
            ax.legend()
            archivo_grafica = os.path.join(CARPETA_GRAFICAS, f"{codigo_estacion}_{col}_eliminados.png")
            plt.savefig(archivo_grafica)
            plt.close()
            print(f"Gráfica guardada en {archivo_grafica}.")
        else:
            print(f"No hay valores eliminados en {col} para graficar.")
    else:
        print(f"No hay datos eliminados en {col}.")

print("\nProceso completado.")