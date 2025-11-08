import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

print("=== Versión optimizada cargada: 5 reps, 30 epochs, batch 128. Sin warnings. ===")

# Configuración
data_path = 'datos_siar_baleares'
estaciones = ['IB01', 'IB02', 'IB03', 'IB04', 'IB05']
output_file = os.path.join(data_path, 'nn_errors_fast.csv')

# Combinaciones de inputs (según Tabla 4 del TFG)
input_combinations = {
    'ANN_Rs': ['Radiacion', 'TempMedia'],  # 2 inputs
    'ANN_Ra': ['TempMax', 'TempMin', 'TempMedia', 'Ra'],  # 4 inputs
    'ANN_HR': ['TempMax', 'TempMin', 'TempMedia', 'Ra', 'HumedadMedia']  # 5 inputs
}

# Función para calcular métricas
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    aare = np.mean(np.abs((y_true - y_pred) / y_true)) if np.all(y_true != 0) else np.nan
    return mse