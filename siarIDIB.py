import pandas as pd



# Ruta al archivo Excel corregida
archivo_entrada = 'C:/Users/josep/OneDrive/Documentos/GitHub/Capitulo-1/estaciones_listado.xlsx'  # Cambia esto si tu archivo tiene otro nombre

# Cargar el archivo Excel en un DataFrame (usa read_excel para .xlsx)
df = pd.read_excel(archivo_entrada)

# Filtrar estaciones cuyo código empieza por "IB"
baleares_df = df[df['Codigo'].str.startswith('IB')]

# Mostrar el número de estaciones encontradas
print(f"Se encontraron {len(baleares_df)} estaciones en las Islas Baleares.")

# Mostrar la lista filtrada (opcional, para verificar)
print(baleares_df)

# Guardar el resultado en un nuevo CSV
archivo_salida = 'C:/Users/josep/OneDrive/Documentos/GitHub/Capitulo-1/estaciones_baleares.csv'
baleares_df.to_csv(archivo_salida, index=False)
print(f"Archivo guardado en: {archivo_salida}")