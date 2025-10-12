import os
import pandas as pd

data_path = r'C:/Users/josep/OneDrive/Documentos/GitHub/Capitulo-1/datos_siar_baleares'

# Check catálogo
cat_file = os.path.join(data_path, 'estaciones_baleares.csv')
if os.path.exists(cat_file):
    df_cat = pd.read_csv(cat_file)
    print("Columnas en estaciones_baleares.csv:", df_cat.columns.tolist())
    if 'IB05' in df_cat['Codigo'].values:
        print("IB05 encontrado en catálogo! Fila:")
        print(df_cat[df_cat['Codigo'] == 'IB05'])
    else:
        print("IB05 NO en catálogo.")
else:
    print("Catálogo no encontrado.")

# Check datos depurados
data_file = os.path.join(data_path, 'IB05_datos_depurados.csv')
if os.path.exists(data_file):
    print(f"{data_file} EXISTE. Primeras 3 filas:")
    df_data = pd.read_csv(data_file, nrows=3)
    print(df_data)
else:
    print(f"{data_file} NO existe. Verifica nombre exacto (case-sensitive).")

# Lista todos los archivos IB* para ver
print("Archivos en carpeta que empiezan con IB:")
print([f for f in os.listdir(data_path) if f.startswith('IB') and f.endswith('.csv')])