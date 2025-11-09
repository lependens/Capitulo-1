import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import warnings
# Suprimir warnings durante la carga de datos (común en entornos de despliegue)
warnings.filterwarnings('ignore') 

# Inicializar la app con tema Bootstrap para diseño moderno
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# 🚨 ESTA LÍNEA ES CRUCIAL PARA EL DESPLIEGUE ONLINE (Gunicorn/Render) 🚨
server = app.server 

# =========================================================================
# 1. CARGA GLOBAL DE DATOS
# =========================================================================
# Definir la ruta base como el directorio donde reside este script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas prioritarias. En un entorno online, el servidor debe tener esta estructura:
# /app_dash_online.py
# /datos_siar_baleares/IBXX_et0_variants_ajustado.csv
data_path_priority = os.path.join(SCRIPT_DIR, 'datos_siar_baleares')
data_path_fallback = SCRIPT_DIR # Por si los archivos están en la raíz

# Estaciones que el código busca
estaciones = ['IB01', 'IB02', 'IB03', 'IB04', 'IB05'] 

# Estilos y configuración
font_style = {'family': 'Arial, sans-serif', 'size': 14, 'color': '#333333'}
header_style = {'fontWeight': 'bold', 'color': '#2c3e50', 'marginBottom': '10px'}
table_style = {'textAlign': 'left', 'fontFamily': 'Arial', 'fontSize': '14px'}

# Definición extendida de modelos para el dashboard
MODELOS_DICT = {
    'ET0_calc': {'label': 'Penman-Monteith (SIAR)', 'color': '#007bff'},
    'ET0_sun': {'label': 'Sunshine', 'color': '#1abc9c'},
    'ET0_harg': {'label': 'Hargreaves Original', 'color': '#f39c12'},
    'ET0_val': {'label': 'Valiantzas Original', 'color': '#e74c3c'},
    'ET0_harg_ajustado': {'label': 'Hargreaves Ajustado', 'color': '#3498db'},
    'ET0_val_ajustado': {'label': 'Valiantzas Ajustado', 'color': '#2ecc71'},
}

MODELO_REFERENCIA = 'ET0_calc'
MODELOS_EMPIRICOS = [col for col in MODELOS_DICT.keys() if col != MODELO_REFERENCIA]


def load_data_globally():
    """Carga y concatena todos los archivos de datos de ET0, intentando varias rutas."""
    df_all = pd.DataFrame()
    found_estaciones = []
    
    print("Iniciando carga global de archivos CSV...")

    paths_to_check = [data_path_priority, data_path_fallback]
    
    for path in paths_to_check:
        estaciones_a_buscar = [code for code in estaciones if code not in found_estaciones]

        for code in estaciones_a_buscar:
            filename = f'{code}_et0_variants_ajustado.csv'
            filepath = os.path.join(path, filename)
            
            try:
                if not os.path.exists(filepath):
                    continue

                df = pd.read_csv(filepath)
                df['Estacion'] = code
                df['Fecha'] = pd.to_datetime(df['Fecha'])
                
                # Convertir columnas clave a numérico
                cols_to_convert = list(MODELOS_DICT.keys()) + ['TempMedia', 'Radiacion', 'HumedadMedia']
                for col in cols_to_convert:
                    # Usar 'errors=coerce' para reemplazar valores no numéricos con NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df_all = pd.concat([df_all, df], ignore_index=True)
                found_estaciones.append(code)
                print(f"✅ Archivo {filename} cargado exitosamente desde {path}.")

            except Exception as e:
                print(f"ERROR: No se pudo procesar el archivo {filename} en {path}. Error: {e}")

        if len(found_estaciones) == len(estaciones):
            break
        
    if df_all.empty:
        print("¡ERROR CRÍTICO! No se pudo cargar ningún dato.")
        return pd.DataFrame(), []
    
    # Añadir columna de mes para análisis estacional
    if not df_all.empty:
        df_all['Mes'] = df_all['Fecha'].dt.month
        # Quitar NaNs y valores extremos de ET0 de la referencia (PM)
        df_all = df_all[df_all[MODELO_REFERENCIA].notna() & (df_all[MODELO_REFERENCIA] >= 0)]

    
    return df_all, found_estaciones

df_global, found_estaciones_global = load_data_globally()

# =========================================================================
# 2. CONFIGURACIÓN INICIAL (Solo si hay datos cargados)
# =========================================================================
if df_global.empty:
    error_message = html.Div(
        [
            html.H1("⚠️ Error Crítico de Carga de Datos", style={'color': '#e74c3c'}),
            html.P("La aplicación no pudo encontrar o procesar ningún archivo CSV de datos."),
            html.P("Asegúrate de que los archivos IBXX_et0_variants_ajustado.csv estén en una carpeta llamada 'datos_siar_baleares' en tu repositorio."),
        ], style={'textAlign': 'center', 'marginTop': '50px', 'fontFamily': 'Arial'}
    )
    app.layout = dbc.Container(error_message)
else:
    min_year = df_global['Fecha'].dt.year.min()
    max_year = df_global['Fecha'].dt.year.max()
    years = list(range(min_year, max_year + 1))
    
    # Columnas para la tabla de errores
    errors_columns = [
        {"name": "Modelo", "id": "Modelo"},
        {"name": "MSE (mm²/d²)", "id": "MSE"},
        {"name": "RRMSE", "id": "RRMSE"},
        {"name": "MAE (mm/día)", "id": "MAE"},
        {"name": "R²", "id": "R2"},
        {"name": "AARE", "id": "AARE"},
    ]
    
    # =========================================================================
    # 3. FUNCIONES DE CÁLCULO DE MÉTRICAS Y GRÁFICOS
    # =========================================================================

    def calculate_errors_df(df_filtered):
        """Calcula las métricas de error para todos los modelos empíricos vs PM."""
        if df_filtered.empty:
            return pd.DataFrame()

        cols_to_use = [MODELO_REFERENCIA] + MODELOS_EMPIRICOS
        df_comp = df_filtered[cols_to_use].dropna()
        
        if df_comp.empty:
            return pd.DataFrame()

        y_true = df_comp[MODELO_REFERENCIA]
        mean_y_true = y_true.mean()
        
        results = []
        for col in MODELOS_EMPIRICOS:
            model_label = MODELOS_DICT[col]['label']
            y_pred = df_comp[col]
            
            # Métricas
            mse = ((y_true - y_pred) ** 2).mean()
            rmse = np.sqrt(mse)
            mae = (np.abs(y_true - y_pred)).mean()
            
            # Cálculo de R2 (ajustado a la implementación clásica)
            numerator_r2 = np.sum((y_true - y_pred) ** 2)
            denominator_r2 = np.sum((y_true - mean_y_true) ** 2)
            r2 = 1 - (numerator_r2 / denominator_r2) if denominator_r2 != 0 else 0
            
            # Cálculo de AARE
            aare = (np.abs((y_true - y_pred) / y_true.replace(0, np.nan))).mean()
            
            results.append({
                "Modelo": model_label,
                "MSE": f"{mse:.3f}",
                "RRMSE": f"{rmse / mean_y_true:.3f}" if mean_y_true else "N/A",
                "MAE": f"{mae:.3f}",
                "R2": f"{r2:.3f}",
                "AARE": f"{aare:.3f}",
            })
            
        return pd.DataFrame(results)
        
    def create_scatter_fig(df_filtered, selected_model_col):
        """Crea el gráfico de nube de puntos (Scatter Plot) de ET0_calc vs Modelo Empírico."""
        if df_filtered.empty or selected_model_col == MODELO_REFERENCIA:
            return go.Figure()

        df_plot = df_filtered[[MODELO_REFERENCIA, selected_model_col]].dropna().copy()
        
        y_true = df_plot[MODELO_REFERENCIA]
        y_pred = df_plot[selected_model_col]
        
        # Función auxiliar para calcular R2 (necesaria para el scatter plot)
        def r2_score(y_true, y_pred):
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        r2 = r2_score(y_true, y_pred) if len(y_true) > 1 and y_true.std() > 0 else 0
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        
        fig = go.Figure()

        # Puntos de dispersión (diarios)
        fig.add_trace(go.Scattergl( # Usamos Scattergl para datasets grandes
            x=y_true, 
            y=y_pred,
            mode='markers',
            marker=dict(
                size=5,
                color=MODELOS_DICT[selected_model_col]['color'],
                opacity=0.6
            ),
            name=f'{MODELOS_DICT[selected_model_col]["label"]} (R²: {r2:.3f})'
        ))

        # Línea de ajuste 1:1 (referencia)
        fig.add_trace(go.Scatter(
            x=[min_val*0.9, max_val*1.1], 
            y=[min_val*0.9, max_val*1.1],
            mode='lines',
            line=dict(color='black', dash='dash', width=2),
            name='Línea 1:1'
        ))
        
        fig.update_layout(
            title=f'Correlación Diaria: {MODELOS_DICT[selected_model_col]["label"]} vs PM',
            xaxis_title=f'ET₀ Penman-Monteith (mm/día)',
            yaxis_title=f'ET₀ {MODELOS_DICT[selected_model_col]["label"]} (mm/día)',
            font=font_style, 
            hovermode="closest",
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='#f8f9fa'
        )
        
        fig.update_xaxes(range=[min_val*0.9, max_val*1.1], showgrid=True, gridcolor='#eaeaea')
        fig.update_yaxes(range=[min_val*0.9, max_val*1.1], showgrid=True, gridcolor='#eaeaea')
        
        return fig
        
    def create_monthly_diff_fig(df_filtered, selected_model_col):
        """Crea el gráfico de la diferencia media mensual (Modelo - PM)."""
        if df_filtered.empty or selected_model_col == MODELO_REFERENCIA:
            return go.Figure()
        
        df_monthly = df_filtered.groupby('Mes')[[MODELO_REFERENCIA, selected_model_col]].mean().reset_index()
        
        df_monthly['Diferencia_Media'] = df_monthly[selected_model_col] - df_monthly[MODELO_REFERENCIA]
        
        meses = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun', 
                 7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'}
        df_monthly['Mes_Label'] = df_monthly['Mes'].map(meses)

        fig = px.bar(
            df_monthly, 
            x='Mes_Label', 
            y='Diferencia_Media', 
            color='Diferencia_Media',
            color_continuous_scale=px.colors.diverging.RdBu, # Escala divergente para mostrar errores positivos/negativos
            labels={'Diferencia_Media': 'Error Medio (mm/día)', 'Mes_Label': 'Mes'},
            title=f'Error Medio Mensual: {MODELOS_DICT[selected_model_col]["label"]} - Penman-Monteith'
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="grey")
        
        fig.update_layout(
            font=font_style, 
            coloraxis_showscale=False,
            title_font_size=18,
            plot_bgcolor='white',
            paper_bgcolor='#f8f9fa'
        )
        
        return fig

    def create_all_models_time_series(df_filtered):
        """
        Crea un gráfico de series temporales para TODOS los modelos empíricos seleccionados vs PM.
        """
        if df_filtered.empty:
            return go.Figure()

        # Tomamos una muestra para que el gráfico no sea demasiado pesado en el navegador
        # Tomar 1 de cada 100 puntos si el dataset es muy grande, o 1 de cada 10 para datasets medianos.
        # Aquí, haremos una media móvil para suavizar y reducir puntos.
        window_size = 7
        df_plot = df_filtered[['Fecha'] + list(MODELOS_DICT.keys())].set_index('Fecha').rolling(window=window_size).mean().reset_index().dropna()
        
        # Tomar solo una muestra de 1 de cada 5 puntos del promedio móvil
        df_plot = df_plot.iloc[::5, :]

        fig = go.Figure()

        for col, settings in MODELOS_DICT.items():
            fig.add_trace(go.Scatter(
                x=df_plot['Fecha'],
                y=df_plot[col],
                mode='lines',
                name=settings['label'],
                line=dict(color=settings['color'], width=2 if col == MODELO_REFERENCIA else 1),
                opacity=1.0 if col == MODELO_REFERENCIA else 0.8
            ))

        fig.update_layout(
            title=f'Series Temporales Suavizadas de ET₀ (Media Móvil de {window_size} días)',
            xaxis_title='Fecha',
            yaxis_title='ET₀ (mm/día)',
            font=font_style,
            hovermode="x unified",
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='#f8f9fa'
        )
        
        fig.update_xaxes(showgrid=True, gridcolor='#eaeaea')
        fig.update_yaxes(showgrid=True, gridcolor='#eaeaea')

        return fig


    # =========================================================================
    # 4. LAYOUT DEL DASHBOARD
    # =========================================================================

    # Dropdowns para el control
    station_options = [{'label': f'Estación {e}', 'value': e} for e in found_estaciones_global]
    model_options = [{'label': v['label'], 'value': k} for k, v in MODELOS_DICT.items() if k != MODELO_REFERENCIA]
    model_options.insert(0, {'label': 'Selecciona Modelo', 'value': 'ET0_harg_ajustado'}) # Default

    controls = dbc.Card(
        [
            html.H4("Controles de Visualización", className="card-title"),
            html.P("Selecciona los filtros para actualizar el dashboard.", className="card-text"),
            
            html.Div(
                [
                    dbc.Label("Estación Meteorológica"),
                    dcc.Dropdown(
                        id='station-dropdown',
                        options=station_options,
                        value=found_estaciones_global[0] if found_estaciones_global else None,
                        clearable=False,
                        style={'fontFamily': 'Arial'}
                    ),
                ], className="mb-3"
            ),
            
            html.Div(
                [
                    dbc.Label("Modelo Empírico a Comparar"),
                    dcc.Dropdown(
                        id='model-dropdown',
                        options=model_options,
                        value='ET0_harg_ajustado',
                        clearable=False,
                        style={'fontFamily': 'Arial'}
                    ),
                ], className="mb-3"
            ),
            
            html.Div(
                [
                    dbc.Label("Rango de Años"),
                    dcc.RangeSlider(
                        id='year-slider',
                        min=min_year,
                        max=max_year,
                        step=1,
                        marks={year: str(year) for year in years if year % 4 == 0 or year == min_year or year == max_year},
                        value=[min_year, max_year]
                    ),
                ], className="mb-3"
            ),
        ],
        body=True,
        className="shadow-sm border-0 bg-light p-4"
    )

    app.layout = dbc.Container(
        [
            html.Div(
                [
                    html.H1("Dashboard de Análisis Comparativo de ET₀", className="text-center text-primary mb-2"),
                    html.P("Comparativa entre Penman-Monteith (SIAR) y modelos empíricos en Baleares.", className="text-center text-muted"),
                    html.Hr(className="my-3")
                ],
            ),
            
            dbc.Row(
                [
                    dbc.Col(controls, md=4, className="mb-4"),
                    dbc.Col(
                        dbc.Alert(
                            id="data-info",
                            color="info",
                            className="text-center"
                        ), 
                        md=8, 
                        className="mb-4"
                    ),
                ],
                align="start"
            ),
            
            html.Div(id='dashboard-content', className="mt-4"),
            
            html.Div(
                id='error-message', 
                style={'color': 'red', 'fontWeight': 'bold', 'textAlign': 'center', 'marginTop': '20px'}
            )

        ],
        fluid=True,
        className="p-5"
    )

    # =========================================================================
    # 5. CALLBACKS DE LA APLICACIÓN
    # =========================================================================

    @app.callback(
        Output('data-info', 'children'),
        Output('dashboard-content', 'children'),
        Output('error-message', 'children'),
        Input('station-dropdown', 'value'),
        Input('model-dropdown', 'value'),
        Input('year-slider', 'value')
    )
    def update_dashboard(selected_station, selected_model, year_range):
        """Filtra datos y genera todos los gráficos y la tabla de errores."""
        error_msg = ""
        
        if not selected_station or not selected_model or not year_range:
            return "Ajusta los filtros para visualizar los datos.", html.Div(), ""

        try:
            min_year_filter, max_year_filter = year_range
            
            # 1. Filtrado de Datos
            df_filtered = df_global[
                (df_global['Estacion'] == selected_station) &
                (df_global['Fecha'].dt.year >= min_year_filter) &
                (df_global['Fecha'].dt.year <= max_year_filter)
            ].copy()
            
            if df_filtered.empty:
                return (
                    f"No hay datos para {selected_station} en el rango {min_year_filter}-{max_year_filter}.",
                    html.Div(),
                    ""
                )
            
            # 2. Resumen de datos y metadatos
            num_rows = len(df_filtered)
            start_date = df_filtered['Fecha'].min().strftime('%Y-%m-%d')
            end_date = df_filtered['Fecha'].max().strftime('%Y-%m-%d')
            
            data_info = (
                f"Estación: {selected_station} | Periodo: {start_date} a {end_date} | "
                f"Días analizados: {num_rows:,}"
            )

            # 3. Generación de Contenido
            
            # Tabla de Errores (calculada para todos los modelos en el periodo filtrado)
            errors_df = calculate_errors_df(df_filtered)
            
            # Gráficos
            fig_scatter = create_scatter_fig(df_filtered, selected_model)
            fig_diff_month = create_monthly_diff_fig(df_filtered, selected_model)
            fig_time = create_all_models_time_series(df_filtered)
            
            content = [
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    html.H3('Métricas de Error (Periodo Filtrado)', style=header_style),
                                    dash_table.DataTable(
                                        id='errors-table',
                                        data=errors_df.to_dict('records'),
                                        columns=errors_columns,
                                        style_cell={**table_style, 'padding': '10px'},
                                        style_header={'fontWeight': 'bold', 'backgroundColor': '#e9ecef'},
                                        style_table={'marginBottom': '20px', 'borderRadius': '5px', 'overflowX': 'auto'},
                                    ),
                                ],
                                className="p-4 bg-white rounded shadow-sm"
                            ), 
                            md=12, 
                            className="mb-4"
                        ),
                    ]
                ),
                
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    html.H3('Correlación Diaria', style=header_style),
                                    html.P(f'Comparación del modelo {MODELOS_DICT[selected_model]["label"]} vs Penman-Monteith.', className="text-muted"),
                                    dcc.Graph(figure=fig_scatter)
                                ],
                                className="p-4 bg-white rounded shadow-sm h-100"
                            ), 
                            md=6, 
                            className="mb-4"
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    html.H3('Error por Mes', style=header_style),
                                    html.P(f'Diferencia media mensual ({MODELOS_DICT[selected_model]["label"]} - PM).', className="text-muted"),
                                    dcc.Graph(figure=fig_diff_month)
                                ],
                                className="p-4 bg-white rounded shadow-sm h-100"
                            ), 
                            md=6, 
                            className="mb-4"
                        ),
                    ]
                ),

                dbc.Row(
                    dbc.Col(
                        html.Div(
                            [
                                html.H3('Series Temporales Suavizadas (Todos los Modelos)', style=header_style),
                                html.P('Muestra el promedio móvil de 7 días de todos los modelos.', className="text-muted"),
                                dcc.Graph(figure=fig_time)
                            ],
                            className="p-4 bg-white rounded shadow-sm"
                        ),
                        md=12
                    )
                )

            ]
            return data_info, html.Div(content), error_msg
        
        except Exception as e:
            error_msg = f"🚨 Error procesando la estación {selected_station}: {str(e)}"
            print(error_msg)
            return (
                f"Error al cargar datos para {selected_station}.",
                html.Div(html.P(f"Ha ocurrido un error inesperado. Consulte la consola para más detalles.")),
                error_msg
            )

    # Solo para ejecución local (no es usado en Render, pero necesario para pruebas locales)
    if __name__ == '__main__':
        print("\n--- Ejecutando Dash localmente ---")
        print(f"Estaciones encontradas: {found_estaciones_global}")
        app.run_server(debug=True, host='0.0.0.0', port=8050)
        
# Mostrar error si no hay datos disponibles
if df_global.empty and 'app' in globals():
    if __name__ == '__main__':
        app.run_server(debug=True, host='0.0.0.0', port=8050)