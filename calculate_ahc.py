
import pandas as pd
import numpy as np
import os

# Ruta a los datos (ajusta si es necesario)
data_path = r'C:/Users/josep/OneDrive/Documentos/GitHub/Capitulo-1/datos_siar_baleares'

# Lista de estaciones (solo IB01-IB05, ya que IB06-IB11 no tienen datos)
estaciones = [f'IB{str(i).zfill(2)}' for i in range(1, 6)]

# Función para calcular AHC y ajustar ET0 (como en MATLAB: mean(ratios), sin 'a')
def calculate_ahc_for_station(code):
    file = os.path.join(data_path, f'{code}_et0_variants.csv')
    if not os.path.exists(file):
        print(f"Archivo no encontrado para {code}: {file}")
        return None, None
    
    df = pd.read_csv(file)
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    df = df.dropna(subset=['Fecha', 'EtPMon', 'ET0_calc', 'ET0_harg', 'ET0_val'])  # Asegurar datos válidos
    
    if df.empty:
        print(f"No datos válidos para {code}")
        return None, None
    
    # Coeficientes diarios (para columnas adicionales)
    df['A_diario_harg'] = df['ET0_calc'] / df['ET0_harg']
    df['A_diario_val'] = df['ET0_calc'] / df['ET0_val']
    
    # AHC como mean(ratios diarios), como en MATLAB
    ahc_harg = np.nanmean(df['A_diario_harg'])  # nanmean para manejar NaNs
    ahc_val = np.nanmean(df['A_diario_val'])
    
    # ET0 ajustados (como en MATLAB: emp * ahc)
    df['ET0_harg_ajustado'] = df['ET0_harg'] * ahc_harg
    df['ET0_val_ajustado'] = df['ET0_val'] * ahc_val
    
    # Guardar CSV actualizado
    output_file = os.path.join(data_path, f'{code}_et0_variants_ajustado.csv')
    df.to_csv(output_file, index=False)
    print(f"Procesado {code}: AHC_harg = {ahc_harg:.4f}, AHC_val = {ahc_val:.4f}. Guardado en {output_file}")
    
    return ahc_harg, ahc_val

# Procesar todas y generar MD
def generate_ahc_md():
    ahc_data = {'Estacion': [], 'AHC_Hargreaves': [], 'AHC_Valiantzas': []}
    ahc_harg_list = []
    ahc_val_list = []
    
    for code in estaciones:
        ahc_harg, ahc_val = calculate_ahc_for_station(code)
        if ahc_harg is not None:
            ahc_data['Estacion'].append(code)
            ahc_data['AHC_Hargreaves'].append(round(ahc_harg, 4))
            ahc_data['AHC_Valiantzas'].append(round(ahc_val, 4))
            ahc_harg_list.append(ahc_harg)
            ahc_val_list.append(ahc_val)
    
    # General (media de AHC por estación)
    if ahc_harg_list:
        ahc_harg_general = np.mean(ahc_harg_list)
        ahc_val_general = np.mean(ahc_val_list)
        ahc_data['Estacion'].append('General')
        ahc_data['AHC_Hargreaves'].append(round(ahc_harg_general, 4))
        ahc_data['AHC_Valiantzas'].append(round(ahc_val_general, 4))
    
    df_ahc = pd.DataFrame(ahc_data)
    
    # Generar MD
    md_content = "### Coeficientes de Ajuste (AHC) por Estación\n\n"
    md_content += df_ahc.to_markdown(index=False)
    
    with open(os.path.join(data_path, 'ahc_por_estacion.md'), 'w', encoding='utf-8') as f:
        f.write(md_content)
    print("Tabla de AHC guardada en 'ahc_por_estacion.md'")

if __name__ == "__main__":
    generate_ahc_md()
