import pandas as pd
import numpy as np
import os

data_path = r'C:/Users/josep/OneDrive/Documentos/GitHub/Capitulo-1/datos_siar_baleares'
estaciones = [f'IB{str(i).zfill(2)}' for i in range(1, 12)]

def calculate_errors(obs, est):
    if len(obs) != len(est) or len(obs) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    mse = np.mean((obs - est)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(obs - est))
    if len(obs) > 1 and np.std(obs) > 0 and np.std(est) > 0:
        r2 = np.corrcoef(obs, est)[0,1]**2
    else:
        r2 = np.nan
    aare = np.mean(np.abs((obs - est) / obs)) if np.all(obs != 0) else np.nan
    return mse, rmse, mae, r2, aare

models = ['ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val']
error_names = ['MSE', 'RMSE', 'MAE', 'R2', 'AARE']
errors = {model: [] for model in models}

for code in estaciones:
    file = os.path.join(data_path, f'{code}_et0_variants.csv')
    if os.path.exists(file):
        df = pd.read_csv(file)
        obs = df['EtPMon'].values
        for model in models:
            if model in df.columns:
                est = df[model].values
                df_errors = calculate_errors(obs, est)
                errors[model].append(df_errors)

# Mean errors
mean_errors = {model: np.nanmean(errors[model], axis=0) if errors[model] else [np.nan] * 5 for model in models}

table_data = pd.DataFrame(mean_errors, index=error_names).transpose()
print(table_data.to_string())