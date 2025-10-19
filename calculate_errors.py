import pandas as pd
import numpy as np
import os

# Ruta a los datos (ajusta si es necesario)
data_path = r'C:/Users/josep/OneDrive/Documentos/GitHub/Capitulo-1/datos_siar_baleares'

# Lista de estaciones (IB01 a IB11)
estaciones = [f'IB{str(i).zfill(2)}' for i in range(1, 12)]

# Función para calcular errores según tus fórmulas
def calculate_errors(obs, est):
    valid = ~np.isnan(obs) & ~np.isnan(est) & (obs != 0)
    obs, est = obs[valid], est[valid]
    if len(obs) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    # MSE: 1/n * sum((x - x_J)^2)
    mse = np.mean((obs - est)**2)
    
    # RMSE: sqrt(MSE) / mean(obs)
    rmse = np.sqrt(mse) / np.mean(obs) if np.mean(obs) != 0 else np.nan
    
    # MAE: 1/n * sum(|x - x_J|)
    mae = np.mean(np.abs(obs - est))
    
    # R²: (sum((x - mean(x))(x_J - mean(x_J))))^2 / (sum((x - mean(x))^2) * sum((x_J - mean(x_J))^2))
    mean_obs = np.mean(obs)
    mean_est = np.mean(est)
    num = np.sum((obs - mean_obs) * (est - mean_est))**2
    denom = np.sum((obs - mean_obs)**2) * np.sum((est - mean_est)**2)
    r2 = num / denom if denom != 0 else np.nan
    
    # AARE: 1/n * sum(|(x - x_J)/x|)
    aare = np.mean(np.abs((obs - est) / obs)) if np.all(obs != 0) else np.nan
    
    return round(mse, 2), round(rmse, 2), round(mae, 2), round(r2, 2), round(aare, 2)

# Modelos a comparar
models = {
    'PM Estándar': 'ET0_calc',
    'PM Cielo Claro': 'ET0_sun',
    'Hargreaves': 'ET0_harg',
    'Valiantzas': 'ET0_val'
}

# Procesar por estación y generar tablas MD
def generate_error_tables():
    # DataFrame general para concatenar todas las estaciones
    df_all = pd.DataFrame()
    
    # Diccionario para almacenar tablas MD
    md_tables = {}
    
    for code in estaciones:
        file = os.path.join(data_path, f'{code}_et0_variants.csv')
        if not os.path.exists(file):
            print(f"Archivo no encontrado para {code}: {file}")
            continue
        
        df = pd.read_csv(file)
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df = df.dropna(subset=['Fecha', 'EtPMon'])  # Asegurar obs válido
        
        obs = df['EtPMon'].values
        error_data = {}
        
        for model_name, col in models.items():
            if col in df.columns:
                est = df[col].values
                error_data[model_name] = calculate_errors(obs, est)
            else:
                error_data[model_name] = (np.nan, np.nan, np.nan, np.nan, np.nan)
        
        # Crear DataFrame de errores para la estación
        error_names = ['MSE (mm²/día²)', 'RMSE (adimensional)', 'MAE (mm/día)', 'R² (adimensional)', 'AARE (adimensional)']
        errors_df = pd.DataFrame(error_data, index=error_names).transpose()
        errors_df = errors_df.reset_index().rename(columns={'index': 'Modelo'})
        
        # Generar tabla MD para la estación
        md_table = f"### Errores para Estación {code}\n\n"
        md_table += "| Modelo | MSE (mm²/día²) | RMSE (adimensional) | MAE (mm/día) | R² (adimensional) | AARE (adimensional) |\n"
        md_table += "|--------|----------------|---------------------|--------------|-------------------|---------------------|\n"
        for _, row in errors_df.iterrows():
            md_table += f"| {row['Modelo']} | {row['MSE (mm²/día²)']} | {row['RMSE (adimensional)']} | {row['MAE (mm/día)']} | {row['R² (adimensional)']} | {row['AARE (adimensional)']} |\n"
        md_tables[code] = md_table
        
        # Concatenar para general
        df['Estacion'] = code  # Añadir columna Estacion para General
        df_all = pd.concat([df_all, df], ignore_index=True)
    
    # Calcular errores para General (media de todas estaciones)
    if not df_all.empty:
        obs_all = df_all['EtPMon'].values
        error_data_all = {}
        
        for model_name, col in models.items():
            if col in df_all.columns:
                est_all = df_all[col].values
                error_data_all[model_name] = calculate_errors(obs_all, est_all)
            else:
                error_data_all[model_name] = (np.nan, np.nan, np.nan, np.nan, np.nan)
        
        errors_df_all = pd.DataFrame(error_data_all, index=error_names).transpose()
        errors_df_all = errors_df_all.reset_index().rename(columns={'index': 'Modelo'})
        
        # Generar tabla MD para General
        md_table_all = "### Errores Generales (Media de Todas Estaciones)\n\n"
        md_table_all += "| Modelo | MSE (mm²/día²) | RMSE (adimensional) | MAE (mm/día) | R² (adimensional) | AARE (adimensional) |\n"
        md_table_all += "|--------|----------------|---------------------|--------------|-------------------|---------------------|\n"
        for _, row in errors_df_all.iterrows():
            md_table_all += f"| {row['Modelo']} | {row['MSE (mm²/día²)']} | {row['RMSE (adimensional)']} | {row['MAE (mm/día)']} | {row['R² (adimensional)']} | {row['AARE (adimensional)']} |\n"
        md_tables['General'] = md_table_all
    
    return md_tables

# Ejecutar y guardar tablas MD
if __name__ == "__main__":
    tables = generate_error_tables()
    
    # Guardar en un archivo .md
    with open(os.path.join(data_path, 'analisis_errores.md'), 'w', encoding='utf-8') as f:
        for key, table in tables.items():
            f.write(table + "\n\n---\n\n")
    print("Tablas guardadas en 'analisis_errores.md'")
    
    # Imprimir en consola para revisión
    for key, table in tables.items():
        print(table)
        print("\n---\n")