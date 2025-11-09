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

    paths_to_check = [data_path_priority]
    if data_path_fallback != data_path_priority:
         paths_to_check.append(data_path_fallback)

    for path in paths_to_check:
        # Copiamos la lista para saber qué buscar en esta iteración
        estaciones_a_buscar = [code for code in estaciones if code not in found_estaciones]

        for code in estaciones_a_buscar:
            filename = f'{code}_et0_variants_ajustado.csv'
            filepath = os.path.join(path, filename)
            
            try:
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"El archivo no existe en la ruta.")

                df = pd.read_csv(filepath)
                df['Estacion'] = code
                df['Fecha'] = pd.to_datetime(df['Fecha'])
                
                # Convertir columnas clave a numérico
                cols_to_convert = list(MODELOS_DICT.keys()) + ['TempMedia', 'Radiacion']
                for col in cols_to_convert:
                    # Usar 'errors=coerce' para reemplazar valores no numéricos con NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df_all = pd.concat([df_all, df], ignore_index=True)
                found_estaciones.append(code)
                print(f"✅ Archivo {filename} cargado exitosamente desde {path}.")

            except FileNotFoundError:
                print(f"Advertencia: Archivo {filename} no encontrado en {path}.")
            except Exception as e:
                print(f"ERROR: No se pudo procesar el archivo {filename} en {path}. Error: {e}")

        # Si encontramos todos los datos, salimos del bucle de rutas
        if len(found_estaciones) == len(estaciones):
            break
        
    if df_all.empty:
        print("¡ERROR CRÍTICO! No se pudo cargar ningún dato.")
        return pd.DataFrame(), []
    
    # Añadir columna de mes para análisis estacional
    if not df_all.empty:
        df_all['Mes'] = df_all['Fecha'].dt.month
    
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
            html.P(f"Por favor, asegúrate de que los archivos {', '.join([f'{e}_et0_variants_ajustado.csv' for e in estaciones])} "
                   f"estén en la carpeta 'datos_siar_baleares' dentro de la carpeta del proyecto."),
        ], style={'textAlign': 'center', 'marginTop': '50px'}
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

        # Usar sólo el subconjunto de columnas necesario para el cálculo
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
            
            # Cálculo de R2
            numerator_r2 = np.sum((y_true - y_pred) ** 2)
            denominator_r2 = np.sum((y_true - mean_y_true) ** 2)
            r2 = 1 - (numerator_r2 / denominator_r2) if denominator_r2 != 0 else 0
            
            # Cálculo de AARE
            aare = (np.abs((y_true - y_pred) / y_true)).replace([np.inf, -np.inf], np.nan).dropna().mean()
            
            results.append({
                "Modelo": model_label,
                "MSE": f"{mse:.3f}",
                "RRMSE": f"{rmse / mean_y_true:.3f}" if mean_y_true else "N/A",
                "MAE": f"{mae:.3f}",
                "R2": f"{r2:.3f}",
                "AARE": f"{aare:.3f}",
            })
            
        return pd.DataFrame(results)
        
    def create_scatter_fig(df_filtered, selected_model_col, station_title):
        """Crea el gráfico de nube de puntos (Scatter Plot) de ET0_calc vs Modelo Empírico."""
        if df_filtered.empty or selected_model_col == MODELO_REFERENCIA:
            return go.Figure()

        df_plot = df_filtered[[MODELO_REFERENCIA, selected_model_col]].dropna().copy()
        
        y_true = df_plot[MODELO_REFERENCIA]
        y_pred = df_plot[selected_model_col]
        
        # Calcular R2 para el título
        r2 = r2_score(y_true, y_pred) if len(y_true) > 1 and y_true.std() > 0 else 0
        
        # Ajuste de la línea 1:1 para el rango de valores
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        
        fig = go.Figure()

        # Puntos de dispersión (diarios)
        fig.add_trace(go.Scatter(
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
            x=[min_val, max_val], 
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='black', dash='dash'),
            name='Línea 1:1'
        ))
        
        # Configuración del layout
        fig.update_layout(
            title=f'Correlación Diaria: {MODELOS_DICT[selected_model_col]["label"]} vs PM',
            xaxis_title=f'ET₀ Penman-Monteith (mm/día)',
            yaxis_title=f'ET₀ {MODELOS_DICT[selected_model_col]["label"]} (mm/día)',
            font=font_style, 
            hovermode="closest",
            showlegend=True
        )
        
        fig.update_xaxes(range=[min_val*0.9, max_val*1.1])
        fig.update_yaxes(range=[min_val*0.9, max_val*1.1])
        
        return fig
        
    def create_monthly_diff_fig(df_filtered, selected_model_col):
        """Crea el gráfico de la diferencia media mensual (Modelo - PM)."""
        if df_filtered.empty or selected_model_col == MODELO_REFERENCIA:
            return go.Figure()
        
        # Agrupar por mes y calcular la media de ambos modelos
        df_monthly = df_filtered.groupby('Mes')[[MODELO_REFERENCIA, selected_model_col]].mean().reset_index()
        
        # Calcular la diferencia media mensual (Modelo - PM)
        df_monthly['Diferencia_Media'] = df_monthly[selected_model_col] - df_monthly[MODELO_REFERENCIA]
        
        meses = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun', 
                 7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'}
        df_monthly['Mes_Label'] = df_monthly['Mes'].map(meses)

        # Crear el gráfico de barras
        fig = px.bar(
            df_monthly, 
            x='Mes_Label', 
            y='Diferencia_Media', 
            color='Diferencia_Media',
            color_continuous_scale=px.colors.sequential.RdBu, # Escala de color para las diferencias
            labels={'Diferencia_Media': 'Error Medio (mm/día)', 'Mes_Label': 'Mes'},
            title=f'Error Medio Mensual: {MODELOS_DICT[selected_model_col]["label"]} - Penman-Monteith'
        )
        
        # Añadir línea de referencia cero
        fig.add_hline(y=0, line_dash="dash", line_color="grey")
        
        fig.update_layout(
            font=font_style, 
            coloraxis_showscale=False,
            title_font_size=18
        )
        
        return fig

    def create_all_models_time_series(df_filtered, station_title):
        """
        NUEVO: Crea un gráfico de series temporales (con puntos) para TODOS los modelos.
        Recomendado solo cuando se filtra una estación específica.
        """
        if df_filtered.empty:
            return go.Figure()

        fig = go.Figure()
        
        # Usamos los datos de Fecha, y todas las columnas ET0 relevantes
        df_plot = df_filtered[['Fecha'] + list(MODELOS_DICT.keys())].dropna()

        # Iterar sobre todos los modelos para agregar trazas
        for col, config in MODELOS_DICT.items():
            # Seleccionamos modo 'lines+markers' para ver la evolución y la densidad de puntos
            fig.add_trace(go.Scatter(
                x=df_plot['Fecha'],
                y=df_plot[col],
                mode='lines+markers',
                marker=dict(size=3, opacity=0.5),
                line=dict(width=1),
                name=config['label'],
                line_color=config['color']
            ))

        fig.update_layout(
            title=f'Serie Temporal Diaria de ET₀: Comparación de Todos los Modelos en {station_title}',
            xaxis_title='Fecha',
            yaxis_title='ET₀ (mm/día)',
            font=font_style,
            hovermode="x unified",
            height=500
        )
        return fig

    # Función auxiliar para calcular R2 (necesaria para el scatter plot)
    def r2_score(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # =========================================================================
    # 4. DISEÑO DE LA APLICACIÓN DASH
    # =========================================================================

    # Nuevo componente de explicación de modelos (Igual al anterior)
    model_explanation_card = dbc.Card([
        dbc.CardHeader("Fórmulas y Contexto de Comparación", style={'backgroundColor': '#2980b9', 'color': 'white', 'fontWeight': 'bold'}),
        dbc.CardBody([
            html.P(html.Strong("Modelo de Referencia (Base de Comparación):"), className="mb-1"),
            html.Div([
                html.Span("Penman-Monteith (SIAR)", style={'color': MODELOS_DICT['ET0_calc']['color'], 'fontWeight': 'bold'}),
                html.Ul([
                    html.Li("Es el modelo estándar FAO-56, considerado el más preciso."),
                    html.Li("Inputs: Temperatura media, Radiación, Humedad, Velocidad del Viento."),
                    html.Li("Contexto: Todos los modelos empíricos se evalúan contra PM (ET0_calc) para determinar su precisión en Baleares."),
                ], className="ml-3"),
            ], className="mb-3 border-bottom pb-2"),

            html.P(html.Strong("Modelos Empíricos (Simples):"), className="mb-1"),
            html.Ul([
                html.Li([
                    html.Span("Sunshine (ET0_sun):", style={'color': MODELOS_DICT['ET0_sun']['color'], 'fontWeight': 'bold'}),
                    " Basado solo en Radiación (Rs) y Temperatura Media."
                ]),
                html.Li([
                    html.Span("Hargreaves Original (ET0_harg):", style={'color': MODELOS_DICT['ET0_harg']['color'], 'fontWeight': 'bold'}),
                    " Basado en TMax, TMin y Radiación Extraterrestre (Ra). Ideal donde faltan datos de viento/humedad."
                ]),
                html.Li([
                    html.Span("Valiantzas Original (ET0_val):", style={'color': MODELOS_DICT['ET0_val']['color'], 'fontWeight': 'bold'}),
                    " Basado en TMax, TMin, Ra y Humedad Media. Mejor para zonas costeras húmedas."
                ]),
            ], className="ml-3"),

            html.P(html.Strong("Modelos Ajustados:"), className="mb-1 mt-3"),
            html.Ul([
                html.Li([
                    html.Span("Hargreaves Ajustado (ET0_harg_ajustado):", style={'color': MODELOS_DICT['ET0_harg_ajustado']['color'], 'fontWeight': 'bold'}),
                    " Hargreaves recalibrado con coeficientes AHC (ajustados a Baleares) para reducir el error sistemático."
                ]),
                html.Li([
                    html.Span("Valiantzas Ajustado (ET0_val_ajustado):", style={'color': MODELOS_DICT['ET0_val_ajustado']['color'], 'fontWeight': 'bold'}),
                    " Valiantzas recalibrado con coeficientes AHC (ajustados a Baleares) para obtener una mejor correlación."
                ]),
            ], className="ml-3"),

        ], style={'fontSize': '13px', 'lineHeight': '1.5'}),
    ], className="mb-4 shadow-sm")


    app.layout = dbc.Container([
        dbc.Row(dbc.Col(html.H1("Dashboard de Análisis Comparativo de ET₀", 
                                className="text-center my-4", 
                                style={'color': '#1abc9c', 'fontFamily': 'Arial, sans-serif'}))),
        
        dbc.Row([
            # Panel de Control (4 columnas)
            dbc.Col([
                # Card de Explicación de Modelos
                model_explanation_card,

                html.Div([
                    html.Label("1. Seleccionar Estación (para filtrar datos):", style=header_style),
                    dcc.Dropdown(
                        id='dropdown-estacion',
                        options=[{'label': 'Todas las Estaciones (Media)', 'value': 'ALL'}] + 
                                [{'label': f'Estación {e}', 'value': e} for e in found_estaciones_global],
                        value='IB01', # Valor inicial en IB01 para mostrar la serie temporal por defecto
                        clearable=False,
                        style=font_style
                    ),
                ], className="mb-4"),
                
                html.Div([
                    html.Label("2. Seleccionar Rango de Años:", style=header_style),
                    dcc.RangeSlider(
                        id='slider-year',
                        min=min_year,
                        max=max_year,
                        value=[min_year, max_year],
                        marks={str(y): str(y) for y in years if y % 5 == 0 or y == min_year or y == max_year},
                        step=1,
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ], className="mb-4 pt-3"),
                
                html.Hr(),

                html.Div([
                    html.Label("3. Modelo a Comparar (vs PM) - En Nube de Puntos y Errores:", style=header_style),
                    dcc.Dropdown(
                        id='dropdown-model-comp',
                        options=[{'label': v['label'], 'value': k} for k, v in MODELOS_DICT.items() if k != MODELO_REFERENCIA],
                        value='ET0_val_ajustado', # Seleccionar el mejor por defecto
                        clearable=False,
                        style=font_style
                    ),
                ], className="mb-4"),
                
                # Nuevo filtro para AHC (Solo tiene efecto si se selecciona un modelo ajustado)
                html.Div([
                    html.Label("4. Estación de Origen de AHC (Análisis de Transferibilidad):", 
                               style={**header_style, 'fontSize': '14px', 'fontWeight': 'normal'}),
                    dcc.Dropdown(
                        id='dropdown-ahc-estacion',
                        options=[{'label': f'AHC de {e} (Usado en el CSV)', 'value': e} for e in found_estaciones_global],
                        value=found_estaciones_global[0], # Valor inicial
                        clearable=False,
                        style={'fontSize': '12px'}
                    ),
                    html.Small("Nota: Actualmente, los datos CSV solo contienen el AHC original de cada estación, por lo que este selector solo es un placeholder de visualización y no cambia el AHC interno del modelo.", className="text-muted"),
                ], className="mb-4 p-3 border rounded"),


            ], md=4, className="p-4 rounded-3 shadow-lg", style={'backgroundColor': '#ecf0f1'}), # Fondo más claro para el panel de control
            
            # Panel de Salida (8 columnas)
            dbc.Col(html.Div(id='output-dashboard'), md=8),
        ], className="my-4"),
        
        html.Div(id='data-error-message', style={'color': '#e74c3c', 'fontWeight': 'bold', 'textAlign': 'center', 'marginTop': '20px'}),

    ], fluid=True, style={'backgroundColor': '#f5f7f9', 'minHeight': '100vh'}) # Fondo general más claro


    # =========================================================================
    # 5. CALLBACKS
    # =========================================================================

    @app.callback(
        [Output('output-dashboard', 'children'),
         Output('data-error-message', 'children')],
        [Input('dropdown-estacion', 'value'),
         Input('slider-year', 'value'),
         Input('dropdown-model-comp', 'value'),
         Input('dropdown-ahc-estacion', 'value')] # Aunque no se usa directamente en el cálculo, se pasa para el título.
    )
    def update_dashboard(code, year_range, selected_model_col, ahc_station):
        error_msg = ""
        
        # 1. Filtrar los datos globales
        try:
            start_year, end_year = year_range
            
            # Filtrar por rango de años en el dataset global
            df_year_filtered = df_global[
                (df_global['Fecha'].dt.year >= start_year) & 
                (df_global['Fecha'].dt.year <= end_year)
            ].copy()

            if df_year_filtered.empty:
                error_msg = f"🚨 No hay datos disponibles en el rango {start_year}-{end_year}."
                return html.Div(error_msg), error_msg
            
            # Filtrar por estación o usar todos para el análisis
            if code == 'ALL':
                df_filtered = df_year_filtered
                station_title = "Media Agregada de Todas las Estaciones"
                show_time_series = False
            elif code in found_estaciones_global:
                df_filtered = df_year_filtered[df_year_filtered['Estacion'] == code]
                station_title = f"Estación {code}"
                show_time_series = True
            else:
                error_msg = f"🚨 La estación seleccionada ({code}) no tiene datos cargados."
                return html.Div(error_msg), error_msg

            if df_filtered.empty:
                error_msg = f"🚨 No hay datos disponibles para {station_title} en el rango {start_year}-{end_year}."
                return html.Div(error_msg), error_msg

            # 2. Generar Componentes
            
            # Cálculo de Errores para la selección actual (Estación o ALL)
            errors_df_current = calculate_errors_df(df_filtered)
            errors_df_global = calculate_errors_df(df_year_filtered) 
            
            # Gráfico de Nube de Puntos (Scatter Plot)
            fig_scatter = create_scatter_fig(df_filtered, selected_model_col, station_title)
            
            # Gráfico de Diferencias Mensuales
            fig_monthly_diff = create_monthly_diff_fig(df_filtered, selected_model_col)
            
            # Gráfico de Serie Temporal (solo si no es ALL)
            if show_time_series:
                 fig_time_series = create_all_models_time_series(df_filtered, station_title)
                 graph_time_series = dbc.Card([
                    dbc.CardHeader(f"Evolución Diaria de ET₀ de TODOS los Modelos en {station_title}", 
                                   style={'backgroundColor': '#2c3e50', 'color': 'white'}),
                    dbc.CardBody(dcc.Graph(figure=fig_time_series)),
                ], className="mb-4 shadow-sm")
            else:
                graph_time_series = None
            
            # 3. Construir el Layout de Salida
            
            # Título dinámico
            ahc_note = f" (AHC referencia: {ahc_station})" if 'ajustado' in selected_model_col else ""
            
            dashboard_title = html.H2(station_title, 
                                      style={'color': '#2c3e50', 'marginTop': '0', 'marginBottom': '10px', 'textAlign': 'center'})
            
            comparison_subtitle = html.H4(f"Análisis de Correlación y Error Estacional: {MODELOS_DICT[selected_model_col]['label']}{ahc_note} vs Penman-Monteith",
                                          style={'color': MODELOS_DICT[selected_model_col]['color'], 'textAlign': 'center', 'marginBottom': '20px'})


            # Nube de Puntos (Gráfico de Correlación)
            graph_scatter = dbc.Card([
                dbc.CardHeader(f"Nube de Puntos Diarios: {MODELOS_DICT[selected_model_col]['label']} vs PM", 
                               style={'backgroundColor': '#2c3e50', 'color': 'white'}),
                dbc.CardBody(dcc.Graph(figure=fig_scatter)),
            ], className="mb-4 shadow-sm")

            # Gráfico de Diferencias Mensuales (Mensualización del error)
            graph_monthly_diff = dbc.Card([
                dbc.CardHeader(f"Error Medio Mensual (Modelo - PM)", 
                               style={'backgroundColor': '#2c3e50', 'color': 'white'}),
                dbc.CardBody(dcc.Graph(figure=fig_monthly_diff)),
            ], className="mb-4 shadow-sm")
            
            # Tabla de la Estación/ALL seleccionada
            table_current_errors = dbc.Card([
                dbc.CardHeader(f"Métricas de Error para {station_title} (vs Penman-Monteith)", 
                               style={'backgroundColor': '#2c3e50', 'color': 'white'}),
                dbc.CardBody(
                     dash_table.DataTable(
                        id='error-table-current',
                        data=errors_df_current.to_dict('records'),
                        columns=errors_columns,
                        style_cell={**table_style, 'border': '1px solid #ddd', 'minWidth': '100px', 'width': '100px', 'maxWidth': '100px'},
                        style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa', 'color': '#2c3e50'},
                        style_table={'overflowX': 'auto', 'marginBottom': '0px', 'borderRadius': '0 0 0.25rem 0.25rem'},
                    ),
                ),
            ], className="mb-4 shadow-sm")
            
            # Tabla de Media Global
            table_global_errors = dbc.Card([
                dbc.CardHeader(f"Media de Errores en Todas las Estaciones ({', '.join(found_estaciones_global)}) (vs PM)", 
                               style={'backgroundColor': '#1abc9c', 'color': 'white'}),
                dbc.CardBody(
                     dash_table.DataTable(
                        id='error-table-global',
                        data=errors_df_global.to_dict('records'),
                        columns=errors_columns,
                        style_cell={**table_style, 'border': '1px solid #ddd', 'minWidth': '100px', 'width': '100px', 'maxWidth': '100px'},
                        style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa', 'color': '#2c3e50'},
                        style_table={'overflowX': 'auto', 'marginBottom': '0px', 'borderRadius': '0 0 0.25rem 0.25rem'},
                    ),
                ),
            ], className="mb-4 shadow-lg border-success")

            # Construcción del contenido
            content = [dashboard_title]
            
            # 1. Gráfico de Serie Temporal (si aplica)
            if graph_time_series:
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
    # La versión de Dash instalada usa app.run() en lugar de app.run_server()
    print("Iniciando servidor de desarrollo local de Dash...")
    app.run(debug=True)