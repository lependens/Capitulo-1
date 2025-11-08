# Código optimizado - Copia en nueva celda
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Configuración acelerada
data_path = 'datos_siar_baleares'
estaciones = ['IB01', 'IB02', 'IB03', 'IB04', 'IB05']
output_file = os.path.join(data_path, 'nn_errors_fast.csv')  # Nuevo archivo para esta versión

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

# Crear modelo (sin warnings)
def create_model(input_dim, n_neurons):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(n_neurons, activation='tanh'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

# Entrenamiento con k-fold por años (acelerado: 5 repeticiones, 50 epochs)
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
            X_train = train_df[inputs].values
            y_train = train_df['ET0_calc'].values
            X_val = val_df[inputs].values
            y_val = val_df['ET0_calc'].values
            X_test = test_df[inputs].values
            y_test = test_df['ET0_calc'].values
            
            # Normalizar
            X_train_scaled, y_train_scaled, scaler_X, scaler_y = normalize_data(X_train, y_train)
            X_val_scaled = scaler_X.transform(X_val)
            y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))
            X_test_scaled = scaler_X.transform(X_test)
            
            # Entrenar para 1-10 neuronas, 5 repeticiones (acelerado)
            best_val_mse = float('inf')
            best_test_mse = float('inf')
            best_val_model = None
            best_test_model = None
            
            for n_neurons in range(1, 11):
                print(f"    Neurona {n_neurons}:", end=' ')
                for rep in range(1, 6):  # 5 repeticiones (reducido)
                    print(f"rep {rep}", end='.')
                    model = create_model(len(inputs), n_neurons)
                    early_stopping = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)
                    model.fit(
                        X_train_scaled, y_train_scaled,
                        validation_data=(X_val_scaled, y_val_scaled),
                        epochs=50, batch_size=64, verbose=0,  # 50 epochs, batch 64 (acelerado)
                        callbacks=[early_stopping]
                    )
                    
                    # Evaluar validación
                    y_val_pred_scaled = model.predict(X_val_scaled, verbose=0)
                    y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled)
                    val_mse = mean_squared_error(y_val, y_val_pred)
                    
                    if val_mse < best_val_mse:
                        best_val_mse = val_mse
                        best_val_model = model
                        print("v", end='')  # Mejor val
                    
                    # Evaluar test
                    y_test_pred_scaled = model.predict(X_test_scaled, verbose=0)
                    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
                    test_mse = mean_squared_error(y_test, y_test_pred)
                    
                    if test_mse < best_test_mse:
                        best_test_mse = test_mse
                        best_test_model = model
                        print("t", end='')  # Mejor test
                    else:
                        print(".", end='')
                
                print(f" [Mejor val MSE: {best_val_mse:.3f}, Mejor test MSE: {best_test_mse:.3f}]")
            
            # Calcular métricas finales
            for selection, model in [('Validation', best_val_model), ('Test', best_test_model)]:
                y_test_pred_scaled = model.predict(X_test_scaled, verbose=0)
                y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
                mse, rmse, mae, r2, aare = calculate_metrics(y_test, y_test_pred)
                
                results.append({
                    'Estacion': estacion,
                    'Modelo': model_name,
                    'Seleccion': selection,
                    'Test_Year': test_year,
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'AARE': aare
                })
                print(f"    {selection}: MSE {mse:.3f}, MAE {mae:.3f}")
    
    print(f"Estación {estacion} completada.")

# Guardar resultados
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)
print(f"\nResultados guardados en {output_file}")

# Resumen por modelo y estación (media de métricas por año)
summary = results_df.groupby(['Estacion', 'Modelo', 'Seleccion'])[['MSE', 'RMSE', 'MAE', 'R2', 'AARE']].mean().reset_index()
print("\nResumen de métricas promedio:")
print(summary.round(3))

# Guardar resumen
summary.to_csv(os.path.join(data_path, 'nn_errors_summary.csv'), index=False)
print(f"Resumen guardado en {data_path}/nn_errors_summary.csv")