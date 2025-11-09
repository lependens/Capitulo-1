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
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Análisis ET₀ Baleares")
server = app.server  # Esencial para Render/Gunicorn

# =========================================================================
# 1. CARGA GLOBAL DE DATOS (La Corrección Principal para Render)
# =========================================================================
# La carpeta donde deberían estar los archivos CSV (asumiendo estructura local)
data_path = 'datos_siar_baleares' 
# Lista de estaciones a intentar cargar (extensible)
estaciones = ['IB01', 'IB02', 'IB03', 'IB04', 'IB05'] 
data_file_suffix = '_et0_variants_ajustado.csv' # Sufijo de tus archivos

font_style = {'family': 'Arial, sans-serif', 'size': 14, 'color': '#333333'}
header_style = {'fontWeight': 'bold', 'color': '#2c3e50', 'marginBottom': '10px', 'marginTop': '20px', 'textAlign': 'center'}
table_style = {'textAlign': 'left', 'fontFamily': 'Arial', 'fontSize': '14px'}

def calculate_metrics(df):
    """Calcula las métricas de error (MSE, MAE, R2, AARE) para todos los modelos."""
    metrics = []
    
    # Modelos a evaluar contra la referencia SIAR (ET0_calc)
    models_to_evaluate = [
        'ET0_calc', # PM Estándar (debería dar casi 0 error)
        'ET0_sun',  # PM Cielo Claro
        'ET0_harg', # Hargreaves
        'ET0_val',  # Valiantzas
        'ET0_harg_ajustado', # Hargreaves Ajustado
        'ET0_val_ajustado'   # Valiantzas Ajustado
    ]
    
    # Referencia es el valor de PM de SIAR (EtPMon)
    y_true = df['EtPMon'].dropna()
    
    for model in models_to_evaluate:
        # Asegurarse de que solo se usan días con valor de referencia y del modelo
        df_temp = df[[model, 'EtPMon']].dropna()
        if df_temp.empty:
            continue

        y_true_m = df_temp['EtPMon']
        y_pred_m = df_temp[model]

        if len(y_true_m) > 1:
            mse = np.round(mean_squared_error(y_true_m, y_pred_m), 4)
            mae = np.round(mean_absolute_error(y_true_m, y_pred_m), 4)
            r2 = np.round(r2_score(y_true_m, y_pred_m), 4)
            
            # Cálculo de AARE
            diff_abs_rel = np.abs((y_true_m - y_pred_m) / y_true_m)
            aare = np.round(diff_abs_rel[np.isfinite(diff_abs_rel)].mean(), 4) # Manejar división por cero

            # Cálculo de RRMSE
            rmse = np.sqrt(mse)
            mean_y_true = y_true_m.mean()
            rrmse = np.round(rmse / mean_y_true, 4) if mean_y_true != 0 else np.nan

            metrics.append({
                'Modelo': model,
                'Estación': df['Estacion'].iloc[0] if not df.empty else 'N/A',
                'MSE (mm²/día²)': f"{mse:.4f}",
                'RRMSE': f"{rrmse:.4f}",
                'MAE (mm/día)': f"{mae:.4f}",
                'R²': f"{r2:.4f}",
                'AARE': f"{aare:.4f}"
            })

    return pd.DataFrame(metrics)

def load_data_globally():
    """Carga y concatena todos los archivos de datos de ET0 y calcula métricas."""
    df_all = pd.DataFrame()
    errors_df_all = pd.DataFrame()
    found_estaciones = []
    
    print("Iniciando carga global de archivos CSV...")
    
    # Simulación de carga de archivos (en Render, se usa la subida de archivos)
    for code in estaciones:
        filename = f'{code}{data_file_suffix}'
        try:
            # En un entorno real como Render/Canvas, el archivo debe ser accesible
            # Si el archivo está subido (como IB02_et0_variants_ajustado.csv), se usa.
            df_temp = pd.read_csv(filename)
            df_temp['Fecha'] = pd.to_datetime(df_temp['Fecha'])
            df_temp.set_index('Fecha', inplace=True)
            df_temp['Estacion'] = code
            
            # Si 'EtPMon' no existe, lo renombramos de la columna 'ET0_calc' (si existe)
            if 'EtPMon' not in df_temp.columns and 'ET0_calc' in df_temp.columns:
                 # Asumiendo que SIAR proporciona EtPMon. Si no está, la tomamos de nuestro cálculo PM
                 df_temp['EtPMon'] = df_temp['ET0_calc'] 
                 
            # Si 'EtPMon' NO existe, saltamos la estación (o llenamos con NaNs)
            if 'EtPMon' not in df_temp.columns:
                print(f"Advertencia: No se encontró la columna de referencia 'EtPMon' en {filename}. Saltando.")
                continue

            # Cálculo de métricas para la estación y anexión
            errors_df_temp = calculate_metrics(df_temp)
            errors_df_all = pd.concat([errors_df_all, errors_df_temp], ignore_index=True)
            
            df_all = pd.concat([df_all, df_temp])
            found_estaciones.append(code)
            print(f"Cargado {filename}. Filas: {len(df_temp)}")

        except FileNotFoundError:
            print(f"Advertencia: Archivo {filename} no encontrado.")
        except Exception as e:
            print(f"Error al procesar {filename}: {e}")

    # Ordenar por fecha y estación
    if not df_all.empty:
        df_all.sort_values(by=['Estacion', 'Fecha'], inplace=True)
    
    return df_all, errors_df_all, found_estaciones

# Carga de datos al inicio
df_global, errors_df_global, available_estaciones = load_data_globally()

if df_global.empty:
    print("¡ERROR CRÍTICO! No se pudo cargar ningún dato.")
    
# Mapeo de nombres de modelos para visualización en el dashboard
MODEL_NAMES = {
    'EtPMon': 'PM FAO-56 (SIAR Ref.)',
    'ET0_calc': 'PM Estándar (Calculado)',
    'ET0_sun': 'PM Cielo Claro',
    'ET0_harg': 'Hargreaves',
    'ET0_val': 'Valiantzas',
    'ET0_harg_ajustado': 'Hargreaves Ajustado',
    'ET0_val_ajustado': 'Valiantzas Ajustado',
}

# Columnas de error para la tabla
ERRORS_COLUMNS = [
    {"name": "Modelo", "id": "Modelo"},
    {"name": "MSE (mm²/día²)", "id": "MSE (mm²/día²)"},
    {"name": "RRMSE", "id": "RRMSE"},
    {"name": "MAE (mm/día)", "id": "MAE (mm/día)"},
    {"name": "R²", "id": "R²"},
    {"name": "AARE", "id": "AARE"},
]

# =========================================================================
# 2. LAYOUT DE LA APLICACIÓN
# =========================================================================

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Div([
            html.H1("Dashboard de Análisis ET₀ Islas Baleares", style={'textAlign': 'center', 'color': '#2c3e50', 'marginTop': '10px'}),
            html.P("Comparación de modelos de Evapotranspiración (ET₀) vs. Referencia Penman-Monteith SIAR.", style={'textAlign': 'center', **font_style}),
            html.Hr(style={'borderColor': '#bdc3c7'}),
        ]), width=12)
    ]),

    # Controles
    dbc.Row([
        dbc.Col(html.Div([
            html.Label("Seleccionar Estación:", style=font_style),
            dcc.Dropdown(
                id='station-dropdown',
                options=[{'label': f'Estación {c}', 'value': c} for c in available_estaciones],
                value=available_estaciones[0] if available_estaciones else None,
                clearable=False,
                style={'fontFamily': 'Arial'}
            ),
        ], style={'marginBottom': '20px'}), md=6),
        
        dbc.Col(html.Div([
            html.Label("Seleccionar Año:", style=font_style),
            dcc.Dropdown(
                id='year-dropdown',
                # Las opciones se llenan dinámicamente con callback
                clearable=True,
                style={'fontFamily': 'Arial'}
            ),
        ], style={'marginBottom': '20px'}), md=6),
    ], className="mb-4"),

    # Área de contenido dinámico
    html.Div(id='content-output'),

    # Mensaje de Error Global
    html.Div(id='global-error-message', style={'color': 'red', 'fontWeight': 'bold', 'textAlign': 'center', 'paddingTop': '20px'}),

], fluid=True, style={'backgroundColor': '#ecf0f1', 'padding': '20px'})


# =========================================================================
# 3. CALLBACKS DE LA APLICACIÓN
# =========================================================================

# Callback para actualizar las opciones de Año
@app.callback(
    Output('year-dropdown', 'options'),
    Input('station-dropdown', 'value')
)
def set_year_options(selected_station):
    if not selected_station or df_global.empty:
        return []
    
    df_station = df_global[df_global['Estacion'] == selected_station]
    years = sorted(df_station.index.year.unique())
    options = [{'label': 'Todos los Años', 'value': 'ALL'}] + [{'label': str(y), 'value': str(y)} for y in years]
    return options

# Callback principal para generar el contenido del dashboard
@app.callback(
    Output('content-output', 'children'),
    Output('global-error-message', 'children'),
    [Input('station-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def update_dashboard(code, year):
    error_msg = ""
    content = []
    
    if df_global.empty or code is None:
        return html.Div("No hay datos cargados para generar el dashboard.", style={'textAlign': 'center', 'color': 'red', 'padding': '50px'}), error_msg

    try:
        # 1. Filtrar por estación
        df_station = df_global[df_global['Estacion'] == code].copy()
        
        if df_station.empty:
            return html.Div(f"No se encontraron datos para la estación {code}.", style={'textAlign': 'center', 'color': 'red', 'padding': '50px'}), error_msg
        
        # 2. Filtrar por año
        df_filtered = df_station.copy()
        if year and year != 'ALL':
            df_filtered = df_station[df_station.index.year == int(year)]
            
        if df_filtered.empty:
            return html.Div(f"No se encontraron datos para el año {year} en la estación {code}.", style={'textAlign': 'center', 'color': 'red', 'padding': '50px'}), error_msg

        # 3. Obtener tabla de errores
        errors_df = errors_df_global[errors_df_global['Estación'] == code].copy()
        
        # Opcional: Recalcular errores si el año está filtrado (más exacto)
        if year and year != 'ALL':
             errors_df = calculate_metrics(df_filtered)

        # Mapeo de nombres en la tabla
        errors_df['Modelo'] = errors_df['Modelo'].map(MODEL_NAMES)

        # 4. Gráfico de Serie Temporal (ET₀ vs Fecha)
        models_cols = [col for col in df_filtered.columns if col in MODEL_NAMES]
        df_plot_time = df_filtered[models_cols].rename(columns=MODEL_NAMES).reset_index().melt(
            id_vars='Fecha', var_name='Modelo', value_name='ET0 (mm/día)'
        )
        
        fig_time = px.line(df_plot_time, x='Fecha', y='ET0 (mm/día)', color='Modelo', 
                           title=f'Serie Temporal de ET₀ - Estación {code} (Año: {year if year != "ALL" else "Todos"})')
        fig_time.update_layout(legend_title_text='Modelos', font=font_style, plot_bgcolor='white')
        fig_time.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
        fig_time.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
        
        # 5. Gráficos de Diferencias (Dispersión vs TempMedia)
        
        # Modelos cuyas diferencias queremos ver (vs EtPMon)
        diff_models = [m for m in models_cols if m != 'EtPMon']
        
        fig_diff_temp = None
        
        df_diff_temp = []
        for model in diff_models:
            # Calcular la diferencia (Error)
            df_filtered[f'Diff_{model}'] = df_filtered[model] - df_filtered['EtPMon']
            # Agregar a la lista para el melt
            df_temp = df_filtered[['TempMedia', f'Diff_{model}']].dropna()
            df_temp = df_temp.rename(columns={f'Diff_{model}': 'Diferencia (Error)'})
            df_temp['Modelo'] = MODEL_NAMES[model]
            df_diff_temp.append(df_temp)
            
        if df_diff_temp:
            df_plot_diff_temp = pd.concat(df_diff_temp)
            fig_diff_temp = px.scatter(df_plot_diff_temp, x='TempMedia', y='Diferencia (Error)', color='Modelo', 
                                       title=f'Diferencia (Error) de ET₀ vs Temperatura Media',
                                       labels={'TempMedia': 'Temperatura Media (°C)'})
            fig_diff_temp.add_hline(y=0, line_dash="dash", line_color="black")
            fig_diff_temp.update_layout(font=font_style, plot_bgcolor='white')
            fig_diff_temp.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
            fig_diff_temp.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')


        # 6. Gráfico de Diferencias Mensuales (Box Plot)
        df_filtered['Month'] = df_filtered.index.month
        
        df_monthly_diffs = []
        for model in diff_models:
            df_filtered[f'Diff_{model}'] = df_filtered[model] - df_filtered['EtPMon']
            df_temp = df_filtered[['Month', f'Diff_{model}']].dropna()
            df_temp = df_temp.rename(columns={f'Diff_{model}': 'Diferencia (Error)'})
            df_temp['Modelo'] = MODEL_NAMES[model]
            df_monthly_diffs.append(df_temp)
            
        fig_diff_month = None
        if df_monthly_diffs:
            df_plot_diff_month = pd.concat(df_monthly_diffs)
            
            fig_diff_month = px.box(df_plot_diff_month, x='Month', y='Diferencia (Error)', color='Modelo', 
                                    title=f'Distribución del Error de ET₀ por Mes',
                                    labels={'Month': 'Mes (1=Ene, 12=Dic)'})
            fig_diff_month.add_hline(y=0, line_dash="dash", line_color="black")
            fig_diff_month.update_layout(font=font_style, plot_bgcolor='white')
            fig_diff_month.update_xaxes(tickmode='linear')


        # Generación del layout final
        content = [
            html.H2(f'Resultados para Estación {code} (Datos: {df_filtered.index.min().year} - {df_filtered.index.max().year})', 
                    style={'textAlign': 'center', 'color': '#2c3e50', 'marginTop': '30px'}),
            
            html.H3('Tabla de Errores (Métricas)', style=header_style),
            html.P(f"Comparación de modelos contra la referencia PM (SIAR) utilizando datos {'filtrados para el año ' + year if year != 'ALL' else 'completos'}.", 
                   style={'textAlign': 'center', **font_style, 'marginBottom': '20px'}),
            
            dash_table.DataTable(
                id='errors-table',
                data=errors_df.to_dict('records'),
                columns=ERRORS_COLUMNS,
                style_cell={**table_style, 'padding': '10px'},
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa', 'border': '1px solid #dee2e6'},
                style_data_conditional=[
                    {'if': {'column_id': 'Modelo', 'filter_query': '{Modelo} contains "PM FAO-56"'}, 'backgroundColor': '#d4edda', 'fontWeight': 'bold'}, # Referencia en verde
                    {'if': {'column_id': 'Modelo', 'filter_query': '{Modelo} contains "Ajustado"'}, 'backgroundColor': '#e8f5e9'}, # Ajustados en verde claro
                    {'if': {'column_id': 'Modelo', 'filter_query': '{Modelo} contains "PM Cielo Claro"'}, 'backgroundColor': '#fff3cd'}, # Cielo Claro en amarillo
                ],
                style_table={'marginBottom': '30px', 'overflowX': 'auto', 'border': '1px solid #dee2e6'},
            ),
            
            html.H3('Serie Temporal de ET₀', style=header_style),
            html.P("Visualización del comportamiento diario de la evapotranspiración de referencia y los modelos.", style={'textAlign': 'center', **font_style}),
            dcc.Graph(figure=fig_time, style={'marginBottom': '30px', 'backgroundColor': 'white'}),
            
            html.H3('Diferencias (Error) vs Temperatura Media', style=header_style),
            html.P("Cómo varía el error de cada modelo en función de la temperatura media del día. La línea horizontal negra es el error cero.", style={'textAlign': 'center', **font_style}),
            dcc.Graph(figure=fig_diff_temp) if fig_diff_temp else html.P('Datos insuficientes para el gráfico.', style={'textAlign': 'center', **font_style, 'color': 'gray'}),
            
            html.H3('Distribución del Error Mensual', style=header_style),
            html.P("Gráficos de caja (Box Plots) mostrando la distribución del error de cada modelo para cada mes. La caja central es el 50% de los datos y la línea horizontal es la mediana.", style={'textAlign': 'center', **font_style}),
            dcc.Graph(figure=fig_diff_month) if fig_diff_month else html.P('Datos insuficientes para el gráfico.', style={'textAlign': 'center', **font_style, 'color': 'gray'}),
        ]
        return content, error_msg
    
    except Exception as e:
        error_msg = f"🚨 Error crítico procesando la estación {code}: {str(e)}"
        print(error_msg)
        return html.Div(html.P(error_msg, style={'color': 'red', 'padding': '50px', 'textAlign': 'center'})), error_msg

# Ejecución (necesario para el entorno)
if __name__ == '__main__':
    # No se usa app.run_server() directamente en entornos como Render,
    # sino que el servidor Gunicorn se conecta a la variable `server`.
    print("Dash app está lista para ser servida por Gunicorn o similar.")