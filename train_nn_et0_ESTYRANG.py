import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

print("=== Script Interactivo para Entrenamiento de ANN ET₀ ===")
print("Este programa entrena ANN_Rs, ANN_Ra, ANN_HR para una estación y rango de neuronas especificado.")
print("Guarda métricas en CSV y el mejor modelo (.h5) por tipo.")

# Configuración base
data_path = 'datos_siar_baleares'
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

# Crear modelo
def create_model(input_dim, n_neurons):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(n_neurons, activation='tanh'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

# Entrenar para una estación y rango de neuronas
def train_for_station(estacion, min_neurons, max_neurons):
    print(f"\n=== Entrenando para estación {estacion} (neuronas {min_neurons}-{max_neurons}) ===")
    
    df = load_data(estacion)
    if df is None:
        return None
    
    df['Year'] = df['Fecha'].dt.year
    years = sorted(df['Year'].unique())
    print(f"Años disponibles: {len(years)} (de {years[0]} a {years[-1]})")
    
    results = []
    best_models = {}  # {model_name: (model, scaler_X, scaler_y, best_mse)}
    
    for model_name, inputs in input_combinations.items():
        print(f"\n--- Modelo {model_name} ({len(inputs)} inputs) ---")
        
        available_inputs = [col for col in inputs if col in df.columns]
        if len(available_inputs) != len(inputs):
            print(f"Advertencia: Faltan columnas {set(inputs) - set(available_inputs)}")
            continue
        
        best_mse = float('inf')
        best_model_data = None
        
        # K-fold por años
        for test_year in years:
            print(f"  Test year: {test_year}")
            
            test_df = df[df['Year'] == test_year]
            train_val_df = df[df['Year'] != test_year]
            train_df = train_val_df.sample(frac=0.85, random_state=42)
            val_df = train_val_df.drop(train_df.index)
            
            X_train = train_df[inputs].values
            y_train = train_df['ET0_calc'].values
            X_val = val_df[inputs].values
            y_val = val_df['ET0_calc'].values
            X_test = test_df[inputs].values
            y_test = test_df['ET0_calc'].values
            
            X_train_scaled, y_train_scaled, scaler_X, scaler_y = normalize_data(X_train, y_train)
            X_val_scaled = scaler_X.transform(X_val)
            y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))
            X_test_scaled = scaler_X.transform(X_test)
            
            # Entrenar para el rango de neuronas
            for n_neurons in range(min_neurons, max_neurons + 1):
                print(f"    Neurona {n_neurons}:", end=' ')
                for rep in range(1, 6):  # 5 repeticiones
                    print(f"rep {rep}", end='.')
                    model = create_model(len(inputs), n_neurons)
                    early_stopping = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)
                    model.fit(
                        X_train_scaled, y_train_scaled,
                        validation_data=(X_val_scaled, y_val_scaled),
                        epochs=30, batch_size=128, verbose=0,
                        callbacks=[early_stopping]
                    )
                    
                    y_test_pred_scaled = model.predict(X_test_scaled, verbose=0)
                    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
                    test_mse, _, _, _, _ = calculate_metrics(y_test, y_test_pred)
                    
                    if test_mse < best_mse:
                        best_mse = test_mse
                        best_model_data = (model, scaler_X, scaler_y)
                        print("t", end='')  # Mejor test
                    else:
                        print(".", end='')
                
                print(f" [Mejor test MSE: {best_mse:.3f}]")
            
            # Métricas para este test_year
            if best_model_data:
                model, scaler_X, scaler_y = best_model_data
                y_test_pred_scaled = model.predict(X_test_scaled, verbose=0)
                y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
                mse, rmse, mae, r2, aare = calculate_metrics(y_test, y_test_pred)
                results.append({
                    'Estacion': estacion,
                    'Modelo': model_name,
                    'Test_Year': test_year,
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'AARE': aare
                })
        
        # Guardar mejor modelo por tipo
        if best_model_data:
            model, scaler_X, scaler_y = best_model_data
            model_path = os.path.join(data_path, f'best_model_{estacion}_{model_name}.h5')
            model.save(model_path)
            print(f"Mejor modelo guardado: {model_path}")
        
        best_models[model_name] = best_model_data
    
    # Guardar resultados
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"\nResultados guardados en {output_file}")
        
        summary = results_df.groupby(['Estacion', 'Modelo'])[['MSE', 'RMSE', 'MAE', 'R2', 'AARE']].mean().reset_index()
        summary_path = os.path.join(data_path, f'n_{estacion}_summary.csv')
        summary.to_csv(summary_path, index=False)
        print(f"Resumen guardado en {summary_path}")
        print("\nResumen de métricas:")
        print(summary.round(3))
    
    return best_models

# Main interactivo
if __name__ == "__main__":
    print("\n--- Entrenamiento Interactivo ANN ET₀ ---")
    estacion = input("Estación (ej. IB01): ").strip().upper()
    
    rango_input = input("Rango de neuronas (ej. 4-10): ").strip()
    min_neurons, max_neurons = map(int, rango_input.split('-'))
    
    print(f"Entrenando para {estacion}, neuronas {min_neurons}-{max_neurons}...")
    
    best_models = train_for_station(estacion, min_neurons, max_neurons)
    
    if best_models:
        print("\nModelos guardados por tipo:")
        for model_name, model_data in best_models.items():
            print(f"- {model_name}: best_model_{estacion}_{model_name}.h5")
    else:
        print("No se entrenaron modelos válidos.")