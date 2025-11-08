import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import itertools

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

# --- Constantes de Entrenamiento ---
NUM_REPETITIONS = 5  # Número de veces que se entrena cada combinación neuronas/modelo
EPOCHS = 100         # Máximo de épocas
BATCH_SIZE = 128     # Tamaño del lote
PATIENCE = 10        # Para EarlyStopping

# Función para calcular métricas
def calculate_metrics(y_true, y_pred, obs_mean):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calcular RRMSE
    rrmse = rmse / obs_mean if obs_mean != 0 else np.nan
    
    # Calcular AARE
    valid_aare = (y_true != 0)
    y_true_aare, y_pred_aare = y_true[valid_aare], y_pred[valid_aare]
    aare = np.mean(np.abs((y_true_aare - y_pred_aare) / y_true_aare))
    
    return round(mse, 3), round(rmse, 3), round(mae, 3), round(r2, 3), round(aare, 3), round(rrmse, 3)

# Función para construir y entrenar el modelo
def build_and_train_model(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, n_neurons):
    model = Sequential([
        Input(shape=(X_train_scaled.shape[1],)),
        Dense(n_neurons, activation='tanh', name='hidden_layer'),
        Dense(1, activation='linear', name='output_layer')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    
    early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    
    # Entrenar el modelo
    history = model.fit(
        X_train_scaled, y_train_scaled, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        verbose=0, # Ocultar output del entrenamiento para limpieza de consola
        validation_data=(X_test_scaled, y_test_scaled),
        callbacks=[early_stop]
    )
    
    # Evaluar en el conjunto de prueba
    test_loss = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
    
    return model, test_loss

# Función principal de entrenamiento por estación
def train_for_station(estacion, min_neurons, max_neurons):
    # Cargar y preparar datos para la estación
    file_path = os.path.join(data_path, f'{estacion}_filtered.csv')
    
    # FIX: Definir output_file y summary_path dentro del scope de la función
    output_file = os.path.join(data_path, f'n_results_{estacion}.csv') 
    summary_path = os.path.join(data_path, f'n_summary_{estacion}.csv')

    if not os.path.exists(file_path):
        print(f"ERROR: Archivo no encontrado para {estacion}: {file_path}")
        return {}

    df_full = pd.read_csv(file_path, encoding='utf-8-sig')
    df_full['Fecha'] = pd.to_datetime(df_full['Fecha'], errors='coerce', dayfirst=True)
    df_full = df_full.dropna(subset=['Fecha']).sort_values(by='Fecha')
    
    # Eliminar filas con NaN en alguna de las columnas relevantes (inputs + output)
    all_cols_needed = list(set(itertools.chain.from_iterable(input_combinations.values()))) + ['EtPMon']
    df = df_full.dropna(subset=all_cols_needed).reset_index(drop=True)

    if df.empty:
        print(f"ADVERTENCIA: DataFrame vacío después de limpiar NaN para {estacion}.")
        return {}
    
    print(f"\nDatos válidos para {estacion}: {len(df)} filas.")
    
    # Preparar el Escalador (MinMaxScaler)
    # Se entrena un escalador por cada set de inputs (para entradas y salidas)
    
    # Definir años de entrenamiento y prueba
    all_years = sorted(df['Fecha'].dt.year.unique())
    # Usar el último año como test, el resto como train
    test_year = all_years[-1] 
    train_years = all_years[:-1]

    best_models = {}
    results = []

    print(f"Año de test: {test_year}")
    
    # Filtrar datos de entrenamiento y prueba
    df_train = df[df['Fecha'].dt.year.isin(train_years)].copy().reset_index(drop=True)
    df_test = df[df['Fecha'].dt.year == test_year].copy().reset_index(drop=True)
    
    if df_train.empty or df_test.empty:
        print("ERROR: No hay suficientes datos para entrenamiento o prueba.")
        return {}
    
    # Iterar sobre las combinaciones de modelos (ANN_Rs, ANN_Ra, ANN_HR)
    for model_name, inputs in input_combinations.items():
        print(f"\nModelo {model_name}:")
        best_model_data = {'MSE': np.inf, 'model': None, 'neurons': None}
        
        # Obtener los datos para este modelo específico
        X_train = df_train[inputs].values
        X_test = df_test[inputs].values
        y_train = df_train['EtPMon'].values.reshape(-1, 1)
        y_test = df_test['EtPMon'].values.reshape(-1, 1)

        # Escaladores (separados para inputs X y output Y)
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        
        # Entrenar escaladores
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_test_scaled = scaler_y.transform(y_test)
        
        # Media de la variable observada para RRMSE y AARE
        obs_mean = y_test.mean() 
        
        # Iterar sobre el rango de neuronas
        for n_neurons in range(min_neurons, max_neurons + 1):
            print(f"  Neurona {n_neurons}:", end=" ")
            
            best_mse_rep = np.inf
            
            # Repetir el entrenamiento varias veces
            for rep in range(1, NUM_REPETITIONS + 1):
                print(f"rep {rep}..", end="")
                
                # Construir, entrenar y evaluar
                model, test_loss = build_and_train_model(
                    X_train_scaled, y_train_scaled, 
                    X_test_scaled, y_test_scaled, 
                    n_neurons
                )

                # Desescalar predicciones para métricas reales
                y_pred_scaled = model.predict(X_test_scaled, verbose=0)
                y_pred_real = scaler_y.inverse_transform(y_pred_scaled).flatten()
                
                # Calcular métricas (usando y_test real, no escalado)
                mse, rmse, mae, r2, aare, rrmse = calculate_metrics(y_test.flatten(), y_pred_real, obs_mean)

                # Guardar resultados detallados
                results.append({
                    'Estacion': estacion,
                    'Modelo': model_name,
                    'Neuronas': n_neurons,
                    'Repeticion': rep,
                    'Test_Year': test_year,
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'AARE': aare,
                    'RRMSE': rrmse
                })

                # Actualizar el mejor modelo para este número de neuronas
                if mse < best_mse_rep:
                    best_mse_rep = mse
                    
                # Actualizar el mejor modelo global para este tipo de ANN
                if mse < best_model_data['MSE']:
                    best_model_data['MSE'] = mse
                    best_model_data['model'] = model
                    best_model_data['neurons'] = n_neurons
                    
            print(f" [Mejor test MSE: {round(best_mse_rep, 3)}]")

        # Guardar el mejor modelo de este tipo (ANN_Rs, ANN_Ra, o ANN_HR)
        if best_model_data['model'] is not None:
            model_path = os.path.join(data_path, f'best_model_{estacion}_{model_name}.h5')
            # Advertencia: HDF5 es legacy, pero lo mantenemos por compatibilidad con el TFG.
            best_model_data['model'].save(model_path)
            print(f"Mejor modelo guardado: {model_path} (Neuronas: {best_model_data['neurons']}, MSE: {best_model_data['MSE']})")
        
        best_models[model_name] = best_model_data
    
    # Guardar resultados
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"\nResultados detallados guardados en {output_file}")
        
        # Generar y guardar resumen (promedio por Modelo y Neuronas)
        summary = results_df.groupby(['Estacion', 'Modelo', 'Neuronas'])[['MSE', 'RMSE', 'MAE', 'R2', 'AARE', 'RRMSE']].mean().reset_index()
        summary_path_model = os.path.join(data_path, f'n_summary_{estacion}_model.csv')
        summary.to_csv(summary_path_model, index=False)
        print(f"Resumen por Modelo/Neurona guardado en {summary_path_model}")
        
        # Resumen general (promedio por Modelo)
        summary_general = results_df.groupby(['Estacion', 'Modelo'])[['MSE', 'RMSE', 'MAE', 'R2', 'AARE', 'RRMSE']].mean().reset_index()
        summary_general_path = os.path.join(data_path, f'n_summary_{estacion}.csv')
        summary_general.to_csv(summary_general_path, index=False)
        print(f"Resumen general guardado en {summary_general_path}")
        print("\nResumen de métricas (promedio de 5 repeticiones por modelo):")
        print(summary_general.round(3))
    
    return best_models

# Main interactivo
if __name__ == "__main__":
    print("\n--- Entrenamiento Interactivo ANN ET₀ ---")
    
    # Usamos un bucle para manejar errores de entrada
    while True:
        try:
            estacion = input("Estación (ej. IB01): ").strip().upper()
            if not estacion:
                raise ValueError("La estación no puede estar vacía.")
            break
        except Exception as e:
            print(f"Entrada no válida: {e}. Inténtalo de nuevo.")
            
    while True:
        try:
            rango_input = input("Rango de neuronas (ej. 4-10): ").strip()
            min_neurons, max_neurons = map(int, rango_input.split('-'))
            if min_neurons > max_neurons or min_neurons <= 0:
                raise ValueError("Rango inválido. Asegúrate de que min <= max y min > 0.")
            break
        except Exception as e:
            print(f"Formato de rango no válido (debe ser X-Y): {e}. Inténtalo de nuevo.")

    
    print(f"Entrenando para {estacion}, neuronas {min_neurons}-{max_neurons}...")
    
    try:
        train_for_station(estacion, min_neurons, max_neurons)
    except Exception as e:
        print(f"\n🚨 ERROR CRÍTICO durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()