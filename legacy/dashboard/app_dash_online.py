import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore') # Opcional: para limpiar logs de warnings de pandas/numpy

# Inicializar la app con tema Bootstrap para diseño moderno
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Esencial para Render/Gunicorn

# =========================================================================
# 1. CARGA GLOBAL DE DATOS
# =========================================================================
# Definir la ruta base como el directorio donde reside este script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas prioritarias, relativas al directorio del script
data_path_priority = os.path.join(SCRIPT_DIR, 'datos_siar_baleares')
data_path_fallback = SCRIPT_DIR

estaciones = ['IB01', 'IB02', 'IB03', 'IB04', 'IB05'] # Extensible

font_style = {'family': 'Arial, sans-serif', 'size': 14, 'color': '#333333'}
header_style = {'fontWeight': 'bold', 'color': '#2c3e50', 'marginBottom': '10px'}
table_style = {'textAlign': 'left', 'fontFamily': 'Arial', 'fontSize': '14px'}

def load_data_globally():
    """Carga, concatena y corrige (calcula columnas faltantes) todos los archivos de datos de ET0."""
    df_all = pd.DataFrame()
    found_estaciones = []
    
    # Determinar la ruta correcta
    data_path = data_path_priority if os.path.exists(data_path_priority) else data_path_fallback
    print(f"Iniciando carga global de archivos CSV...")
    print(f"Buscando archivos en: {data_path}")

    for code in estaciones:
        file_name = f'{code}_et0_variants_ajustado.csv'
        file_path = os.path.join(data_path, file_name)
        
        if not os.path.exists(file_path):
            file_path = os.path.join(SCRIPT_DIR, file_name) # Buscar en el directorio raíz por si acaso
            if not os.path.exists(file_path):
                # print(f"Archivo no encontrado: {file_name}")
                continue

        try:
            df = pd.read_csv(file_path, low_memory=False)
            df['Fecha'] = pd.to_datetime(df['Fecha'])
            df['Mes'] = df['Fecha'].dt.month
            df['Estacion'] = code
            
            # =================================================================
            # FIX CRÍTICO: Cálculo de columnas de diferencia faltantes (Error: diff_val_ajustado)
            # El error sugiere que el CSV no contiene estas columnas, las calculamos aquí.
            # =================================================================
            if 'ET0_calc' in df.columns:
                
                # 1. Diferencia para Hargreaves Ajustado
                if 'diff_harg_ajustado' not in df.columns and 'ET0_harg_ajustado' in df.columns:
                    df['diff_harg_ajustado'] = df['ET0_calc'] - df['ET0_harg_ajustado']
                
                # 2. Diferencia para Valiantzas Ajustado (CAUSA DEL ERROR)
                if 'diff_val_ajustado' not in df.columns and 'ET0_val_ajustado' in df.columns:
                    df['diff_val_ajustado'] = df['ET0_calc'] - df['ET0_val_ajustado']
                
                # Nota: Las columnas 'diff', 'diff_sun', 'diff_harg', 'diff_val' ya están en el CSV cargado
                
            # =================================================================
            
            df_all = pd.concat([df_all, df], ignore_index=True)
            found_estaciones.append(code)
            print(f"Cargado exitosamente: {file_name}")

        except Exception as e:
            print(f"Error cargando {file_name}: {e}")

    if not df_all.empty:
        df_all.dropna(subset=['ET0_calc'], inplace=True)
        # Convertir a float para cálculos robustos
        for col in ['ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val', 'ET0_harg_ajustado', 'ET0_val_ajustado', 
                    'diff', 'diff_sun', 'diff_harg', 'diff_val', 'diff_harg_ajustado', 'diff_val_ajustado', 
                    'TempMedia', 'Radiacion']:
            if col in df_all.columns:
                # Usar pd.to_numeric con errors='coerce' para manejar posibles valores no numéricos
                df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
        
        # Eliminar filas donde las columnas críticas se convirtieron en NaN
        df_all.dropna(subset=['ET0_calc', 'TempMedia', 'Radiacion'], inplace=True)

    print(f"Carga global finalizada. Estaciones con datos: {found_estaciones}")
    return df_all, found_estaciones

# Carga de datos al inicio
df_data, found_estaciones = load_data_globally()

if df_data.empty:
    print("FATAL: No se cargó ningún dato. Verifique las rutas de los archivos CSV.")
    # Estructura mínima de error para Dash
    app.layout = html.Div([
        html.H1("Error de Carga de Datos", style={'color': 'red'}),
        html.P("No se encontraron o no se pudieron cargar los archivos CSV. Verifique la carpeta 'datos_siar_baleares'.")
    ])
else:
    # Generar la tabla de errores global una sola vez
    def calculate_global_errors(df):
        """Calcula las métricas de error globales (agregadas) para todos los modelos."""
        
        MODELS = {
            'PM Estándar': 'ET0_calc',
            'PM Cielo Claro': 'ET0_sun',
            'Hargreaves': 'ET0_harg',
            'Valiantzas': 'ET0_val',
            'Hargreaves Ajustado': 'ET0_harg_ajustado',
            'Valiantzas Ajustado': 'ET0_val_ajustado',
        }
        
        errors = []
        # La referencia es siempre ET0_calc (PM Estándar)
        y_true = df['ET0_calc']

        for model_name, col_et0 in MODELS.items():
            if col_et0 not in df.columns:
                continue

            y_pred = df[col_et0]
            
            # Filtrar NaN's para un cálculo limpio y evitar warnings de sklearn
            valid_indices = y_true.notna() & y_pred.notna()
            y_true_clean = y_true[valid_indices]
            y_pred_clean = y_pred[valid_indices]
            
            if y_true_clean.empty or model_name == 'PM Estándar':
                # El modelo PM Estándar es la referencia, sus errores son 0.
                if model_name == 'PM Estándar':
                     errors.append({
                        'Modelo': model_name, 
                        'MSE (mm²/día²)': 0.0, 
                        'RRMSE': 0.0, 
                        'MAE (mm/día)': 0.0, 
                        'R²': 1.0, 
                        'AARE': 0.0
                    })
                continue
            
            mse = np.mean((y_true_clean - y_pred_clean)**2)
            rmse = np.sqrt(mse)
            rrmse = rmse / y_true_clean.mean()
            mae = np.mean(np.abs(y_true_clean - y_pred_clean))
            r2 = 1 - (np.sum((y_true_clean - y_pred_clean)**2) / np.sum((y_true_clean - y_true_clean.mean())**2))
            aare = np.mean(np.abs(y_true_clean - y_pred_clean) / y_true_clean)

            errors.append({
                'Modelo': model_name,
                'MSE (mm²/día²)': mse,
                'RRMSE': rrmse,
                'MAE (mm/día)': mae,
                'R²': r2,
                'AARE': aare
            })

        errors_df = pd.DataFrame(errors)
        return errors_df.round(3)

    df_errors_global = calculate_global_errors(df_data)


    # =========================================================================
    # 2. DEFINICIÓN DEL LAYOUT DEL DASHBOARD
    # =========================================================================

    MODELS_DISPLAY = {
        'PM Estándar': 'diff',
        'PM Cielo Claro': 'diff_sun',
        'Hargreaves': 'diff_harg',
        'Valiantzas': 'diff_val',
        'Hargreaves Ajustado': 'diff_harg_ajustado',
        'Valiantzas Ajustado': 'diff_val_ajustado',
    }

    # Opciones para los Dropdowns
    estacion_options = [{'label': f'Estación {code}', 'value': code} for code in found_estaciones]
    model_options = [{'label': name, 'value': name} for name in MODELS_DISPLAY.keys()]

    app.layout = dbc.Container([
        dbc.Row(dbc.Col(html.H1("Análisis de Evapotranspiración (ET₀) en Islas Baleares", className="text-center my-4 text-primary"))),
        
        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id='estacion-dropdown',
                    options=estacion_options,
                    value=found_estaciones[0] if found_estaciones else None,
                    placeholder="Selecciona una Estación",
                    clearable=False,
                    className="mb-3"
                ), md=6
            ),
            dbc.Col(
                dcc.Dropdown(
                    id='modelo-dropdown',
                    options=model_options,
                    value='Valiantzas Ajustado', # Valor por defecto para iniciar la vista
                    placeholder="Selecciona un Modelo a Comparar",
                    clearable=False,
                    className="mb-3"
                ), md=6
            ),
        ]),
        
        # Contenedor para el contenido dinámico del dashboard
        dbc.Card(dbc.CardBody(html.Div(id='dashboard-content')), className="shadow-lg mb-4"),
        
        # Espacio para mensajes de error (impresos en consola)
        html.Div(id='error-output', style={'display': 'none'}),
        
    ], fluid=True)


    # =========================================================================
    # 3. CALLBACKS (Lógica de la Aplicación)
    # =========================================================================

    @app.callback(
        [Output('dashboard-content', 'children'),
         Output('error-output', 'children')],
        [Input('estacion-dropdown', 'value'),
         Input('modelo-dropdown', 'value')]
    )
    def update_dashboard(code, model_name):
        error_msg = ""
        
        if not code or not model_name:
            return html.P("Selecciona una estación y un modelo para empezar.", className="text-center p-5"), error_msg

        try:
            # 1. Filtrar datos por estación
            df_estacion = df_data[df_data['Estacion'] == code].copy()
            if df_estacion.empty:
                error_msg = f"🚨 Error: No se encontraron datos para la estación {code}."
                print(error_msg)
                return html.Div(error_msg, className="alert alert-warning"), error_msg

            # 2. Definir columnas clave
            # Usamos la columna 'diff' del modelo seleccionado
            diff_col = MODELS_DISPLAY.get(model_name)
            et0_col = [key for key, val in MODELS_DISPLAY.items() if val == diff_col][0].replace(' Ajustado', '').replace(' ', '_').lower().replace('pm_estándar', 'et0_calc').replace('pm_cielo_claro', 'et0_sun')
            et0_col = [col_name for col_name in df_estacion.columns if col_name.lower().endswith(et0_col)][0] if et0_col.startswith('et0') else 'ET0_' + diff_col.split('_')[-1] # Intento más robusto

            # Revertir mapeo para obtener el nombre de la columna ET0
            if model_name == 'PM Estándar':
                col_et0_val = 'ET0_calc'
            elif model_name == 'PM Cielo Claro':
                col_et0_val = 'ET0_sun'
            else:
                # Esto busca la columna ET0_harg_ajustado, ET0_val, etc.
                col_et0_val = model_name.replace(' ', '_').replace('Estándar', 'calc').replace('Cielo Claro', 'sun').replace('Hargreaves', 'harg').replace('Valiantzas', 'val').replace('_', '_').lower()
                col_et0_val = 'ET0_' + col_et0_val.split('_')[-1]
                if 'ajustado' in model_name:
                    col_et0_val += '_ajustado'
            
            # Verificación crítica de las columnas antes de graficar
            required_cols = ['Fecha', 'ET0_calc', col_et0_val, diff_col, 'TempMedia', 'Radiacion', 'Mes']
            for col in required_cols:
                if col not in df_estacion.columns:
                    error_msg = f"🚨 Error crítico al actualizar el dashboard: 🚨 Columna de diferencia '{diff_col}' (Modelo '{model_name}') no encontrada en los datos."
                    print(error_msg)
                    return html.Div(error_msg, className="alert alert-danger"), error_msg

            # 3. Cálculo de errores para la estación actual
            df_errors_current = calculate_global_errors(df_estacion)


            # =================================================================
            # 4. Generación de Gráficos
            # =================================================================
            
            # --- Serie Temporal ---
            fig_time = px.line(df_estacion, x='Fecha', y=['ET0_calc', col_et0_val], 
                               title=f'Serie Temporal de ET₀ (PM vs {model_name})',
                               labels={'ET0_calc': 'PM Estándar (Referencia)', col_et0_val: model_name, 'value': 'ET₀ (mm/día)'},
                               color_discrete_map={'ET0_calc': 'darkblue', col_et0_val: 'red'})
            fig_time.update_layout(height=400, template='plotly_white')
            fig_time.update_xaxes(title_text='Fecha')
            fig_time.update_yaxes(title_text='ET₀ (mm/día)')
            graph_time_series = dcc.Graph(figure=fig_time)

            # --- Scatter de Dispersión (PM vs Modelo Seleccionado) ---
            fig_scatter = px.scatter(df_estacion, x='ET0_calc', y=col_et0_val,
                                     title=f'Dispersión: PM vs {model_name} (R²: {df_errors_current[df_errors_current["Modelo"] == model_name]["R²"].iloc[0] if model_name != "PM Estándar" else 1.000})',
                                     labels={'ET0_calc': 'ET₀ PM Estándar (mm/día)', col_et0_val: f'ET₀ {model_name} (mm/día)'},
                                     color_discrete_sequence=['#3498db'])
            
            # Añadir línea 1:1
            max_val = max(df_estacion[['ET0_calc', col_et0_val]].max().max(), 10)
            fig_scatter.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', name='Línea 1:1', line=dict(color='gray', dash='dash')))

            fig_scatter.update_layout(height=400, template='plotly_white')
            graph_scatter = dcc.Graph(figure=fig_scatter)
            

            # --- Diferencias Mensuales (Box Plot) ---
            df_mes = df_estacion.groupby('Mes')[diff_col].agg(['mean', 'std']).reset_index()
            df_mes['Mes_Nombre'] = df_mes['Mes'].apply(lambda x: pd.to_datetime(x, format='%m').strftime('%b'))

            fig_diff_month = px.box(df_estacion, x='Mes', y=diff_col,
                                    title=f'Diferencia Mensual ({model_name} - PM)',
                                    labels={'x': 'Mes', diff_col: 'Diferencia (mm/día)'},
                                    color_discrete_sequence=['#e67e22'])
            fig_diff_month.update_layout(height=400, template='plotly_white')
            fig_diff_month.update_xaxes(tickvals=df_mes['Mes'], ticktext=df_mes['Mes_Nombre'], title_text='Mes')
            graph_monthly_diff = dcc.Graph(figure=fig_diff_month)
            

            # =================================================================
            # 5. Estructura del Contenido
            # =================================================================
            
            dashboard_title = dbc.Alert(
                html.H4(f"Dashboard de Evapotranspiración (ET₀) - Estación {code}", className="alert-heading"),
                color="success",
                className="mb-4 shadow-lg border-success")

            comparison_subtitle = html.H3(f'Análisis de Dispersión y Sesgo para: {model_name}', style=header_style)
            
            # --- Tabla de Errores Estación Actual ---
            errors_columns_current = [{"name": i, "id": i} for i in df_errors_current.columns]
            table_current_errors = html.Div([
                html.H3(f'Métricas de Error para Estación {code} (vs PM Estándar)', style=header_style),
                dash_table.DataTable(
                    id='error-table-current',
                    data=df_errors_current.to_dict('records'),
                    columns=errors_columns_current,
                    style_cell=table_style,
                    style_header={'fontWeight': 'bold', 'backgroundColor': '#e9ecef'},
                    style_table={'marginBottom': '20px'},
                )
            ], className="mb-4")

            # --- Tabla de Errores Global ---
            errors_columns_global = [{"name": i, "id": i} for i in df_errors_global.columns]
            table_global_errors = html.Div([
                html.H3('Métricas de Error Globales (Media de todas las estaciones)', style=header_style),
                dash_table.DataTable(
                    id='error-table-global',
                    data=df_errors_global.to_dict('records'),
                    columns=errors_columns_global,
                    style_cell=table_style,
                    style_header={'fontWeight': 'bold', 'backgroundColor': '#dee2e6'},
                    style_table={'marginBottom': '20px'},
                )
            ], className="mb-4")
            
            # Construcción del contenido
            content = [dashboard_title]
            
            # 1. Gráfico de Serie Temporal (si aplica)
            content.append(graph_time_series)
            
            # 2. Análisis del Modelo Seleccionado
            content.append(comparison_subtitle)
            content.append(dbc.Row([
                dbc.Col(graph_scatter, md=6),
                dbc.Col(graph_monthly_diff, md=6),
            ], className="mb-4"))

            # 3. Tablas de Errores
            content.append(table_current_errors)
            content.append(table_global_errors)
            
            return content, error_msg
        
        except Exception as e:
            error_msg = f"🚨 Error crítico al actualizar el dashboard: {str(e)}"
            print(error_msg)
            # Retorna una vista de error amigable en la app
            return html.Div([
                html.H3("¡Error Inesperado en el Dashboard!", style={'color': '#e74c3c'}),
                html.P(f"El procesamiento falló. Detalle: {str(e)}"),
                html.P("Por favor, revisa la consola para más detalles."),
            ], className="alert alert-danger"), error_msg

if __name__ == '__main__':
    # La versión de Dash instalada usa app.run_server si se ejecuta localmente
    # Render usa 'server' de Gunicorn, así que esta línea es solo para desarrollo local
    app.run_server(debug=True)