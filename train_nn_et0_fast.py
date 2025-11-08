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
    return mse, rmse, mae, r2, aare

# Cargar y preparar datos
def load_data(estacion):
    file = os.path.join(data_path, f'{estacion}_et0_variants.csv')
    if not os.path.exists(file):
        print(f"Archivo no encontrado: {file}")
        return None
    df = pd.read_csv(file, encoding='utf-8-sig', encoding_errors='replace')
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    df = df.dropna(subset=['Fecha', 'ET0_calc'])
    return df

# Normalizar datos
def normalize_data(X, y):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    return X_scaled, y_scaled, scaler_X, scaler_y

# Crear model (sin warning: Input layer explícito)
def create_model(input_dim, n_neurons):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(n_neurons, activation='tanh'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

# Entrenamiento con k-fold por años (acelerado: 5 reps, 30 epochs, batch 128)
results = []
for estacion in estaciones:
    print(f"\n=== Procesando estación {estacion} ===")
    df = load_data(estacion)
    if df is None:
        continue
    
    # Extraer años únicos
    df['Year'] = df['Fecha'].dt.year
    years = sorted(df['Year'].unique())
    print(f"Años disponibles: {len(years)} (de {years[0]} a {years[-1]})")
    
    for model_name, inputs in input_combinations.items():
        print(f"\n--- Modelo {model_name} ({len(inputs)} inputs) ---")
        # Verificar columnas disponibles
        available_inputs = [col for col in inputs if col in df.columns]
        if len(available_inputs) != len(inputs):
            print(f"Advertencia: Faltan columnas {set(inputs) - set(available_inputs)}")
            continue
        
        # K-fold por años (secuencial)
        for test_year in years:
            print(f"  Test year: {test_year} (test: {len(df[df['Year'] == test_year])}, train/val: {len(df[df['Year'] != test_year])})")
            
            # Dividir datos
            test_df = df[df['Year'] == test_year]
            train_val_df = df[df['Year'] != test_year]
            
            # Dividir train/val (85%/15%)
            train_df = train_val_df.sample(frac=0.85, random_state=42)
            val_df = train_val_df.drop(train_df.index)
            
            # Preparar datos
            X_train