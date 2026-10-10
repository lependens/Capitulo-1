import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import joblib 

# =========================================================================
# CONFIGURACIÓN DE TENSORFLOW (SUPRESIÓN DE ADVERTENCIAS Y DISPOSITIVO)
# =========================================================================

# Suprimir advertencias de TensorFlow (incluyendo la de HDF5 legacy format)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def setup_device(use_gpu):
    """Configura TensorFlow para usar CPU o GPU según la elección del usuario."""
    # Deshabilitar explícitamente la GPU para forzar el uso de la CPU por defecto
    tf.config.set_visible_devices([], 'GPU')
    if use_gpu:
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    tf.config.set_visible_devices(gpus[0], 'GPU')
                    print("\n[INFO] Usando la GPU para el entrenamiento.")
                except RuntimeError as e:
                    print(f"\n[ADVERTENCIA] Error al configurar la GPU. Usando CPU: {e}")
            else:
                print("\n[ADVERTENCIA] No se encontraron dispositivos GPU. Usando CPU.")
        except Exception as e:
            print(f"\n[ADVERTENCIA] Error al intentar detectar GPU. Usando CPU: {e}")
    else:
        print("\n[INFO] Usando la CPU para el entrenamiento (Elección del usuario).")

# =========================================================================
# CONFIGURACIÓN GENERAL
# =========================================================================

print("=== Script Unificado para Entrenamiento de ANN ET₀ con CV Flexible ===")

# Configuración base
data_path = 'datos_siar_baleares'
# Estructura de inputs
input_combinations = {
    'ANN_Rs': ['Radiacion', 'TempMedia'],
    'ANN_Ra': ['TempMax', 'TempMin', 'TempMedia', 'Ra'],
    'ANN_HR': ['TempMax', 'TempMin', 'TempMedia', 'Ra', 'HumedadMedia']
}

# Hyperparámetros
EPOCHS = 50  
BATCH_SIZE = 128
NUM_REPS = 5 
PATIENCE = 5 
MIN_TRAIN_YEARS_CAUSAL = 2 # Mínimo de años históricos requeridos para empezar a testear en Forward Chaining.

# =========================================================================
# FUNCIONES DE MÉTRICAS Y MODELO
# =========================================================================

# Función para calcular métricas
def calculate_metrics(y_true, y_pred, y_mean):
    """Calcula MSE, RMSE, MAE, R2, AARE y RRMSE."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Prevenir división por cero en AARE
    y_true_clean = np.where(y_true == 0, 1e-6, y_true)
    y_mean = np.mean(y_true_clean)
    
    mse = mean_squared_error(y_true_clean, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_clean, y_pred)
    r2 = r2_score(y_true_clean, y_pred)
    
    rrmse = rmse / y_mean
    
    # AARE: Average Absolute Relative Error.
    aare = np.mean(np.abs((y_true_clean - y_pred) / y_true_clean))
    
    return mse, rmse, mae, r2, aare, rrmse

# Función para crear y compilar el modelo
def create_model(n_inputs, n_neurons, learning_rate=0.001):
    model = Sequential([
        Input(shape=(n_inputs,)),
        Dense(n_neurons, activation='tanh'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# =========================================================================
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# =========================================================================

def train_and_evaluate_ann(df_full, estacion, min_neurons, max_neurons, cv_type):
    """
    Realiza la Cross-Validation (Causal o No Causal) para la estación.
    """
    
    # Etiqueta para los archivos de salida
    label = "_causal" if cv_type == 'causal' else "_nocausal"
    
    print(f"\n--- FASE 1/5: Preparación de {cv_type.upper()} CV para {estacion} ---")
    
    df = df_full.copy()
    
    all_inputs = list(set([col for inputs in input_combinations.values() for col in inputs]))
    required_cols = all_inputs + ['ET0_calc', 'Año']
        
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=required_cols, inplace=True)
    
    if df.empty:
        print("Error: DataFrame vacío después de limpiar NaNs.")
        return {}

    all_years = sorted(df['Año'].unique())
    
    
    # ========================================================
    # LÓGICA DE VALIDACIÓN CRUZADA (CAUSAL vs. NO CAUSAL)
    # ========================================================
    if cv_type == 'causal':
        # Causal (Forward Chaining): Solo años que tienen suficientes datos previos.
        if len(all_years) < MIN_TRAIN_YEARS_CAUSAL + 1:
            print(f"Error Causal: Se necesitan al menos {MIN_TRAIN_YEARS_CAUSAL + 1} años. Solo hay {len(all_years)}.")
            return {}
            
        cv_years = all_years[MIN_TRAIN_YEARS_CAUSAL:] 
        print(f"[INFO] Método: Causal (Forward Chaining). Años Test: {cv_years}")
        
    else: # cv_type == 'nocausal' (LOYO)
        # No Causal (Leave-One-Year-Out): Todos los años se usan como test.
        cv_years = all_years 
        if len(all_years) < 2:
            print("Error No Causal: Se necesitan al menos 2 años para LOYO.")
            return {}
        print(f"[INFO] Método: No Causal (LOYO). Años Test: {cv_years}")
    
    print(f"[INFO] Total de Años disponibles en el set de datos: {all_years}")
    
    results = []
    best_models = {}

    print("\n--- FASE 2/5: Bucle de Neuronas e Inicialización de Modelos ---")
    
    for n_neurons in range(min_neurons, max_neurons + 1):
        print(f"\n>> EMPEZANDO CON N_NEURONAS: {n_neurons} <<")
        
        for model_name, inputs in input_combinations.items():
            print(f"  --- FASE 3/5: Modelo {model_name} (Inputs: {', '.join(inputs)}) ---")
            
            best_mse_global = np.inf
            best_model_data = None
            
            print(f"  [INFO] Iniciando Validación Cruzada...")
            
            for i, test_year in enumerate(cv_years):
                
                print(f"  > CV Paso {i+1}/{len(cv_years)}: Año Test = {test_year}")
                
                # Definición de sets de entrenamiento según el modo
                if cv_type == 'causal':
                    # Causal: Usa SOLO los años ANTERIORES al año de test.
                    train_years = [y for y in all_years if y < test_year]
                else:
                    # No Causal (LOYO): Usa TODOS los años EXCEPTO el año de test.
                    train_years = [y for y in all_years if y != test_year]
                
                df_train = df[df['Año'].isin(train_years)]
                df_test = df[df['Año'] == test_year]
                
                if df_train.empty:
                    print(f"    ADVERTENCIA: No hay datos de entrenamiento. Saltando Año Test {test_year}.")
                    continue
                
                # 2. Escalado (Ajustado SOLO a los datos de entrenamiento)
                input_scaler = MinMaxScaler(feature_range=(0, 1))
                target_scaler = MinMaxScaler(feature_range=(0, 1))
                
                X_train = df_train[inputs].values
                y_train = df_train['ET0_calc'].values.reshape(-1, 1)
                
                X_test = df_test[inputs].values
                y_test = df_test['ET0_calc'].values.reshape(-1, 1)
                
                if X_test.size == 0:
                    print(f"    ADVERTENCIA: Datos de test insuficientes para Año Test {test_year}. Saltando.")
                    continue
                    
                # Ajustar y transformar datos
                X_train_scaled = input_scaler.fit_transform(X_train)
                y_train_scaled = target_scaler.fit_transform(y_train)
                X_test_scaled = input_scaler.transform(X_test)
                
                y_test_mean = np.mean(y_test)
                
                # 3. Entrenamiento (Monte Carlo Repetitions)
                rep_metrics = []
                
                # Barra de progreso limpia
                sys.stdout.write(f"    FASE 4/5: Repeticiones Monte Carlo ({NUM_REPS}): ")
                sys.stdout.flush()

                for rep in range(1, NUM_REPS + 1):
                    
                    # Crear y entrenar modelo
                    model = create_model(len(inputs), n_neurons)
                    
                    early_stop = EarlyStopping(monitor='loss', patience=PATIENCE, verbose=0, mode='min', restore_best_weights=True)
                    
                    # Entrenamiento
                    model.fit(
                        X_train_scaled, 
                        y_train_scaled, 
                        epochs=EPOCHS, 
                        batch_size=BATCH_SIZE, 
                        verbose=0, 
                        callbacks=[early_stop]
                    )
                    
                    # 4. Evaluación y Desnormalización
                    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
                    y_pred = target_scaler.inverse_transform(y_pred_scaled)
                    
                    # Calcular métricas para esta repetición
                    mse, rmse, mae, r2, aare, rrmse = calculate_metrics(y_test, y_pred, y_test_mean)
                    rep_metrics.append({
                        'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'AARE': aare, 'RRMSE': rrmse
                    })
                    
                    # Guardar el mejor modelo si el MSE de esta repetición es el mejor GLOBAL
                    if mse < best_mse_global:
                        best_mse_global = mse
                        
                        # Rutas de guardado con etiqueta
                        model_path = os.path.join(data_path, f'best_model_{estacion}_{model_name}{label}.keras')
                        input_scaler_path = os.path.join(data_path, f'scaler_input_{estacion}_{model_name}{label}.joblib')
                        target_scaler_path = os.path.join(data_path, f'scaler_target_{estacion}_{model_name}{label}.joblib')
                        
                        model.save(model_path, save_format='keras')
                        joblib.dump(input_scaler, input_scaler_path)
                        joblib.dump(target_scaler, target_scaler_path)
                        
                        best_model_data = {
                            'Neuronas': n_neurons, 
                            'MSE': mse, 
                            'R2': r2, 
                            'Test_Year': test_year,
                            'Path': model_path,
                            'Input_Scaler_Path': input_scaler_path,
                            'Target_Scaler_Path': target_scaler_path
                        }
                    
                    sys.stdout.write(f"[{rep}] ")
                    sys.stdout.flush()
                        
                sys.stdout.write("-> [FIN DE REPETICIONES]\n")
                
                # Promediar métricas de las 5 repeticiones para este año (CV Step)
                avg_metrics = pd.DataFrame(rep_metrics).mean().to_dict()
                print(f"    [RESULTADO] R² (Media Reps): {avg_metrics['R2']:.4f}, RMSE (Media Reps): {avg_metrics['RMSE']:.4f}")
                
                # Añadir resultados al listado general
                results.append({
                    'Estacion': estacion,
                    'Modelo': model_name,
                    'Neuronas': n_neurons,
                    'Año_Test': test_year,
                    'MSE': avg_metrics['MSE'],
                    'RMSE': avg_metrics['RMSE'],
                    'MAE': avg_metrics['MAE'],
                    'R2': avg_metrics['R2'],
                    'AARE': avg_metrics['AARE'],
                    'RRMSE': avg_metrics['RRMSE'],
                    'Metodologia': cv_type.capitalize() # Guardar el tipo de CV
                })
                
            # Almacenar el mejor modelo global encontrado en el rango de neuronas para este tipo de ANN
            print(f"\n  [RESUMEN {model_name}] Mejor MSE global para este modelo/neurona: {best_mse_global:.4f}")
            if best_model_data:
                print(f"  [GUARDADO] Mejor Modelo Global ({model_name}) guardado en: {os.path.basename(best_model_data['Path'])}")
                print(f"  [GUARDADO] Escaladores (Input/Target) guardados en: {os.path.basename(best_model_data['Input_Scaler_Path'])} / {os.path.basename(best_model_data['Target_Scaler_Path'])}")
                
                if model_name not in best_models or best_model_data['MSE'] < best_models[model_name]['MSE']:
                     best_models[model_name] = best_model_data

    
    # 5. Guardar resultados
    print("\n--- FASE 5/5: Guardando Resultados y Resúmenes ---")
    if results:
        results_df = pd.DataFrame(results)
        
        # 5.1 Resultados detallados LOYO 
        output_file = os.path.join(data_path, f'n_results_cv_{estacion}{label}.csv')
        results_df.to_csv(output_file, index=False)
        print(f"[GUARDADO] Resultados detallados (por Año/Rep) en: {output_file}")
        
        # 5.2 Resumen final del CV (promedio de todos los años test y repeticiones)
        summary = results_df.groupby(['Estacion', 'Modelo', 'Neuronas'])[['MSE', 'RMSE', 'MAE', 'R2', 'AARE', 'RRMSE']].mean().reset_index()
        summary_file = os.path.join(data_path, f'n_summary_neuron_{estacion}{label}.csv')
        summary.to_csv(summary_file, index=False)
        print(f"[GUARDADO] Resumen (Media CV por Neurona) en: {summary_file}")

        # 5.3 Resumen general (promedio de todos los años test y neuronas)
        summary_general = results_df.groupby(['Estacion', 'Modelo'])[['MSE', 'RMSE', 'MAE', 'R2', 'AARE', 'RRMSE']].mean().reset_index()
        summary_general_file = os.path.join(data_path, f'n_summary_general_{estacion}{label}.csv')
        summary_general.to_csv(summary_general_file, index=False)
        
        print("\n=== Resumen de Métricas CV (Promedio de CV) ===")
        print(summary_general.round(4))
    
    return best_models

# Main interactivo
if __name__ == "__main__":
    
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    print("\n--- INICIO DEL PROCESO DE ENTRENAMIENTO UNIFICADO ---")
    
    # --- 0. PREGUNTA SOBRE DISPOSITIVO ---
    while True:
        device_choice = input("¿Deseas usar la GPU para el entrenamiento? (s/n): ").strip().lower()
        if device_choice in ['s', 'n']:
            use_gpu_choice = device_choice == 's'
            setup_device(use_gpu_choice)
            break
        else:
            print("Respuesta no válida. Por favor, introduce 's' (sí) o 'n' (no).")
            
    # --- 1. Entrada de Metodología CV ---
    while True:
        cv_choice = input("\nSelecciona la metodología de Validación Cruzada:\n1) No Causal (LOYO/TFG Original)\n2) Causal (Forward Chaining/Estandar Series Temporales)\nOpción (1 o 2): ").strip()
        if cv_choice == '1':
            cv_type = 'nocausal'
            break
        elif cv_choice == '2':
            cv_type = 'causal'
            break
        else:
            print("Opción no válida. Por favor, introduce '1' o '2'.")
            
    # 2. Entrada de Estación
    estacion = input("\nEstación (ej. IB01): ").strip().upper()
    
    # 3. Entrada de Rango de Neuronas
    while True:
        rango_input = input("Rango de neuronas (ej. 4-10, o solo 8): ").strip()
        try:
            if '-' in rango_input:
                min_neurons, max_neurons = map(int, rango_input.split('-'))
            else:
                min_neurons = max_neurons = int(rango_input)
                
            if min_neurons < 1 or max_neurons < min_neurons:
                print("Error: Rango inválido. Asegúrate de que min >= 1 y max >= min.")
                continue
            break
        except ValueError:
            print("Formato de rango no válido (debe ser X-Y o solo X, ej. 4-10 o 10). Inténtalo de nuevo.")
            
    print(f"\n[CONFIG] Estación: {estacion}, Neuronas a probar: {min_neurons} a {max_neurons}")
    print(f"[CONFIG] Método de Validación: {'No Causal (LOYO)' if cv_type == 'nocausal' else 'Causal (Forward Chaining)'}")
    print(f"[CONFIG] {len(input_combinations)} modelos, {NUM_REPS} repeticiones/año.")
    
    # 4. Cargar datos
    file_path = os.path.join(data_path, f'{estacion}_et0_variants_ajustado.csv')
    if not os.path.exists(file_path):
        print(f"\n[CRITICAL ERROR] No se encuentra el archivo de datos ajustados: {file_path}")
        print("Asegúrate de ejecutar el script de cálculo de variantes (variants_et0.py) para generar este archivo.")
        sys.exit(1)
        
    try:
        df_full = pd.read_csv(file_path)
        
        # Procesar Fecha y Año
        df_full['Fecha'] = pd.to_datetime(df_full['Fecha'], errors='coerce', dayfirst=True)
        df_full.dropna(subset=['Fecha'], inplace=True)
        df_full['Año'] = df_full['Fecha'].dt.year
        
        print(f"[DATA] Datos cargados para {estacion}: {len(df_full)} filas.")
        
    except Exception as e:
        print(f"[CRITICAL ERROR] Error al cargar o procesar el archivo CSV: {e}")
        sys.exit(1)

    # 5. Ejecutar el entrenamiento con la metodología elegida
    train_and_evaluate_ann(df_full, estacion, min_neurons, max_neurons, cv_type)

    print("\n--- PROCESO COMPLETADO ---")
    print(f"El entrenamiento ha finalizado. Los archivos de salida están en '{data_path}' y están etiquetados con '_{cv_type}'.")