import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

# Opcional: para limpiar logs de warnings de pandas/numpy
warnings.filterwarnings('ignore')

# Inicializar la app con tema Bootstrap para diseño moderno
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# VARIABLE CRÍTICA para que Render/Gunicorn pueda servir la aplicación
server = app.server

# =========================================================================
# 1. CARGA GLOBAL DE DATOS (Optimización y Preparación para el Servidor)
# =========================================================================
# Definir la ruta base como el directorio donde reside este script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas prioritarias, relativas al directorio del script
DATA_PATH = os.path.join(SCRIPT_DIR, 'datos_siar_baleares')

# Lista de estaciones a intentar cargar (extensible)
ESTACIONES_COD = ['IB01', 'IB02', 'IB03', 'IB04', 'IB05']
DF_ALL = pd.DataFrame() # DataFrame global para almacenar todos los datos cargados
AVAILABLE_ESTACIONES = [] # Estaciones que se pudieron cargar correctamente
AVAILABLE_YEARS = []

# Definición de estilos
FONT_STYLE = {'family': 'Arial, sans-serif', 'size': 14, 'color': '#333333'}
HEADER_STYLE = {'fontWeight': 'bold', 'color': '#2c3e50', 'marginBottom': '10px'}
TABLE_STYLE = {'textAlign': 'left', 'fontFamily': 'Arial', 'fontSize': '14px'}

# Mapping de nombres de variables para los ejes de los gráficos
VAR_MAP = {
    'ET0_calc': 'PM Estándar (ET₀ SIAR)',
    'ET0_val_ajustado': 'Valiantzas Ajustado',
    'ET0_harg_ajustado': 'Hargreaves Ajustado',
    'ET0_sun': 'PM Cielo Claro',
    'TempMedia': 'Temperatura Media (°C)',
    'Radiacion': 'Radiación Solar (MJ/m²)',
}

def load_station_data(code):
    """Carga y procesa los datos para una única estación."""
    filepath = os.path.join(DATA_PATH, f'{code}_et0_variants_ajustado.csv')
    try:
        if not os.path.exists(filepath):
            # Probar en la raíz del script como fallback
            fallback_path = os.path.join(SCRIPT_DIR, f'{code}_et0_variants_ajustado.csv')
            if os.path.exists(fallback_path):
                filepath = fallback_path
            else:
                print(f"⚠️ Archivo no encontrado para {code} en ninguna ruta.")
                return None

        df = pd.read_csv(filepath)
        df['Estacion'] = code
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        df['Mes'] = df['Fecha'].dt.month
        df['Año'] = df['Fecha'].dt.year

        # Calcular diferencias vs. ET0_calc (referencia)
        if 'ET0_calc' in df.columns:
            if 'ET0_harg_ajustado' in df.columns:
                df['diff_harg_ajustado'] = df['ET0_calc'] - df['ET0_harg_ajustado']
            if 'ET0_val_ajustado' in df.columns:
                df['diff_val_ajustado'] = df['ET0_calc'] - df['ET0_val_ajustado']
            if 'ET0_sun' in df.columns:
                df['diff_sun'] = df['ET0_calc'] - df['ET0_sun']

        return df
    except Exception as e:
        print(f"🚨 Error leyendo o procesando {filepath}: {e}")
        return None

def load_data_globally():
    """Carga todos los datos usando multithreading para mayor velocidad."""
    global DF_ALL, AVAILABLE_ESTACIONES, AVAILABLE_YEARS

    print(f"Iniciando carga global de archivos CSV desde {DATA_PATH} o fallback...")

    # Usamos ThreadPoolExecutor para paralelizar la lectura de archivos
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(load_station_data, ESTACIONES_COD))

    loaded_dfs = [res for res in results if res is not None and not res.empty]

    if loaded_dfs:
        DF_ALL = pd.concat(loaded_dfs, ignore_index=True)
        
        # Filtrar solo si las columnas críticas existen
        if 'ET0_calc' in DF_ALL.columns:
             DF_ALL.dropna(subset=['ET0_calc'], inplace=True)
        
        AVAILABLE_ESTACIONES = sorted(DF_ALL['Estacion'].unique().tolist())
        if not DF_ALL.empty and 'Año' in DF_ALL.columns:
            AVAILABLE_YEARS = sorted(DF_ALL['Año'].unique().tolist())
        else:
            AVAILABLE_YEARS = [2000, 2024] # Default range if no data is found

        print(f"✅ Carga global completa. Estaciones disponibles: {AVAILABLE_ESTACIONES}")
        if AVAILABLE_YEARS:
             print(f"Rango de años cargados: {min(AVAILABLE_YEARS)} - {max(AVAILABLE_YEARS)}")
    else:
        print("❌ No se pudo cargar ningún archivo de datos.")
        DF_ALL = pd.DataFrame()
        AVAILABLE_ESTACIONES = []
        AVAILABLE_YEARS = []

# Ejecutar la carga de datos al inicio (CRÍTICO para Render)
load_data_globally()

# =========================================================================
# 2. CÁLCULOS ESTADÍSTICOS Y MÉTODOS DE EVALUACIÓN
# =========================================================================

def r2_score(y_true, y_pred):
    """Implementación de R² para evitar dependencia de sklearn."""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot == 0:
        return np.nan
    return 1 - (ss_res / ss_tot)

def calculate_metrics(y_true, y_pred, model_name, N, station):
    """Calcula y retorna un diccionario con todas las métricas de error."""
    if not y_true.any() or not y_pred.any():
        return None
    
    # Filtrar nulos
    mask = y_true.notna() & y_pred.notna()
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) < 5:
        return None
    
    mse = np.mean((y_true_clean - y_pred_clean)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true_clean - y_pred_clean))
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    # Calcular RMSE relativo y AARE
    mean_true = np.mean(y_true_clean)
    rrmse = (rmse / mean_true) if mean_true != 0 else np.nan
    aare = np.mean(np.abs(y_true_clean - y_pred_clean) / y_true_clean)

    return {
        'Estación': station,
        'Modelo': model_name,
        'N': len(y_true_clean),
        'MSE (mm²/día²)': f'{mse:.4f}',
        'RMSE (mm/día)': f'{rmse:.3f}',
        'RRMSE': f'{rrmse:.3f}',
        'MAE (mm/día)': f'{mae:.3f}',
        'R²': f'{r2:.3f}',
        'AARE': f'{aare:.3f}'
    }

def calculate_errors_for_station(df_station, code):
    """Calcula las métricas de error para todos los modelos en una estación."""
    if df_station.empty or 'ET0_calc' not in df_station.columns:
        return []

    # Columnas esperadas en los datos (deben estar en df_station)
    models = {
        'PM Cielo Claro': 'ET0_sun',
        'Hargreaves Ajustado': 'ET0_harg_ajustado',
        'Valiantzas Ajustado': 'ET0_val_ajustado'
    }
    
    errors = []
    y_true = df_station['ET0_calc']
    N = len(df_station)
    
    for model_name, col_name in models.items():
        if col_name in df_station.columns:
            y_pred = df_station[col_name]
            metrics = calculate_metrics(y_true, y_pred, model_name, N, code)
            if metrics:
                errors.append(metrics)

    return errors

# =========================================================================
# 3. LAYOUT DEL DASHBOARD
# =========================================================================

# Inicializar rango de slider si hay datos
min_year = min(AVAILABLE_YEARS) if AVAILABLE_YEARS else 2000
max_year = max(AVAILABLE_YEARS) if AVAILABLE_YEARS else 2024
marks_dict = {year: str(year) for year in range(min_year, max_year + 1) if year % 4 == 0 or year in [min_year, max_year]}


app.layout = dbc.Container([
    # Título principal
    dbc.Row(dbc.Col(html.H1(
        "ET₀ Islas Baleares: Análisis de Modelos Empíricos",
        className="text-center my-4",
        style={'color': '#007BFF', 'fontFamily': FONT_STYLE['family']}
    ))),

    # Fila de controles (Estación, Año, Modelo, Variable)
    dbc.Row([
        # Dropdown de Estación
        dbc.Col(dbc.Card([
            dbc.CardHeader("Seleccionar Estación"),
            dbc.CardBody(dcc.Dropdown(
                id='station-dropdown',
                options=[{'label': f'{e}', 'value': e} for e in AVAILABLE_ESTACIONES],
                value=AVAILABLE_ESTACIONES[0] if AVAILABLE_ESTACIONES else None,
                clearable=False,
                style=FONT_STYLE
            ))
        ], className="h-100 shadow-sm"), md=3),

        # RangeSlider de Año
        dbc.Col(dbc.Card([
            dbc.CardHeader("Rango de Años"),
            dbc.CardBody(dcc.RangeSlider(
                id='year-slider',
                min=min_year,
                max=max_year,
                value=[min_year, max_year],
                step=1,
                marks=marks_dict,
                tooltip={"placement": "bottom", "always_visible": True}
            ))
        ], className="h-100 shadow-sm"), md=4),
        
        # Dropdown de Modelo para Scatter Plot
        dbc.Col(dbc.Card([
            dbc.CardHeader("Modelo a Comparar"),
            dbc.CardBody(dcc.Dropdown(
                id='model-dropdown',
                options=[
                    {'label': 'Hargreaves Ajustado', 'value': 'ET0_harg_ajustado'},
                    {'label': 'Valiantzas Ajustado', 'value': 'ET0_val_ajustado'},
                    {'label': 'PM Cielo Claro', 'value': 'ET0_sun'},
                ],
                value='ET0_val_ajustado',
                clearable=False,
                style=FONT_STYLE
            ))
        ], className="h-100 shadow-sm"), md=3),
        
        # Dropdown de Variable para Correlación
        dbc.Col(dbc.Card([
            dbc.CardHeader("Variable para Correlación"),
            dbc.CardBody(dcc.Dropdown(
                id='variable-dropdown',
                options=[
                    {'label': 'Temperatura Media', 'value': 'TempMedia'},
                    {'label': 'Radiación Solar', 'value': 'Radiacion'},
                ],
                value='TempMedia',
                clearable=False,
                style=FONT_STYLE
            ))
        ], className="h-100 shadow-sm"), md=2),

    ], className="mb-4"),

    # Fila de Mensajes de Error (al inicio invisible)
    dbc.Row(dbc.Col(html.Div(id='error-message', style={'color': '#e74c3c', 'fontWeight': 'bold'}, children=''))),

    # Contenido dinámico del dashboard
    dbc.Row(dbc.Col(html.Div(id='dashboard-content'))),

], fluid=True)


# =========================================================================
# 4. CALLBACKS: Lógica de Interacción
# =========================================================================

@app.callback(
    [Output('dashboard-content', 'children'),
     Output('error-message', 'children')],
    [Input('station-dropdown', 'value'),
     Input('year-slider', 'value'),
     Input('model-dropdown', 'value'),
     Input('variable-dropdown', 'value')]
)
def update_dashboard(code, year_range, model_y_column, var_x):
    """Genera todos los gráficos y tablas para la estación y rango de años seleccionados."""
    error_msg = ''
    content = html.Div()
    
    if not DF_ALL.empty and code:
        try:
            # 1. Filtrado de datos
            min_year, max_year = year_range
            df_filtered = DF_ALL[
                (DF_ALL['Estacion'] == code) & 
                (DF_ALL['Año'] >= min_year) & 
                (DF_ALL['Año'] <= max_year)
            ]

            if df_filtered.empty or 'ET0_calc' not in df_filtered.columns:
                error_msg = f'⚠️ No hay datos completos (ET₀ PM) para {code} en el rango {min_year}-{max_year}.'
                return html.Div(), error_msg

            # 2. CÁLCULO DE ERRORES (TABLA de la estación actual)
            errors_data = calculate_errors_for_station(df_filtered, code)
            
            if errors_data:
                errors_df = pd.DataFrame(errors_data)
                errors_columns = [{"name": i, "id": i} for i in errors_df.columns]
                
                error_table = dbc.Card([
                    dbc.CardHeader(html.H3(f'Métricas de Evaluación para {code}', style={'color': '#2c3e50', 'fontSize': '1.25rem'})),
                    dbc.CardBody(dash_table.DataTable(
                        id='errors-table-current',
                        data=errors_df.to_dict('records'),
                        columns=errors_columns,
                        style_cell=TABLE_STYLE,
                        style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                        style_table={'overflowX': 'auto'}
                    ))
                ], className="mb-4 shadow-sm border-secondary")
            else:
                error_table = html.P('Datos insuficientes para calcular métricas de error.', className="alert alert-warning")

            
            # 3. GRÁFICO: Serie Temporal de ET₀ (Media Mensual)
            df_monthly = df_filtered.groupby(['Año', 'Mes']).agg(
                {col: 'mean' for col in ['ET0_calc', 'ET0_harg_ajustado', 'ET0_val_ajustado', 'ET0_sun'] if col in df_filtered.columns}
            ).reset_index()
            
            if not df_monthly.empty:
                df_monthly['Fecha_Mensual'] = pd.to_datetime(df_monthly[['Año', 'Mes']].assign(day=1))
                
                # Seleccionar solo las columnas que se pudieron calcular en el agg
                y_cols = [col for col in ['ET0_calc', 'ET0_harg_ajustado', 'ET0_val_ajustado', 'ET0_sun'] if col in df_monthly.columns]

                fig_time = px.line(
                    df_monthly, 
                    x='Fecha_Mensual', 
                    y=y_cols,
                    title=f'Serie Temporal de ET₀ (Media Mensual) en {code}',
                    labels={'value': 'ET₀ (mm/día)', 'Fecha_Mensual': 'Fecha', 'variable': 'Modelo'},
                    color_discrete_map={
                        'ET0_calc': '#007BFF',  # PM Estándar (Referencia)
                        'ET0_harg_ajustado': '#28A745', # Hargreaves Ajustado
                        'ET0_val_ajustado': '#FFC107', # Valiantzas Ajustado
                        'ET0_sun': '#DC3545' # PM Cielo Claro
                    },
                    template='seaborn'
                )
                fig_time.update_layout(legend_title_text='Modelo', font=FONT_STYLE, hovermode="x unified")
                fig_time.for_each_trace(lambda t: t.update(name = VAR_MAP.get(t.name, t.name))) # Renombrar leyenda
                
                graph_time = dbc.Card([
                    dbc.CardHeader(html.H3('Serie Temporal de ET₀ (Media Mensual)', style={'color': '#2c3e50', 'fontSize': '1.25rem'})),
                    dbc.CardBody(dcc.Graph(figure=fig_time))
                ], className="mb-4 shadow-sm border-primary")
            else:
                 graph_time = html.P('Datos insuficientes para la Serie Temporal Mensual.', className="alert alert-warning")


            # 4. GRÁFICO: Scatter Plot de Correlación (PM vs Modelo Seleccionado)
            model_name = [label for opt in app.layout.children[2].children[2].children[1].children[1].options if opt['value'] == model_y_column][0]['label']
            var_x_name = VAR_MAP.get(var_x, var_x)
            
            if model_y_column in df_filtered.columns:
                fig_scatter = go.Figure()
                
                # 4a. Scatter principal: PM vs Modelo Seleccionado
                fig_scatter.add_trace(go.Scatter(
                    x=df_filtered['ET0_calc'], 
                    y=df_filtered[model_y_column], 
                    mode='markers', 
                    name=f'Datos Diarios ({model_name})',
                    marker=dict(color='rgba(0, 123, 255, 0.5)', size=5) # Azul
                ))
                
                # 4b. Línea 1:1 (referencia ideal)
                max_val = max(df_filtered['ET0_calc'].max(), df_filtered[model_y_column].max())
                fig_scatter.add_trace(go.Line(
                    x=[0, max_val], y=[0, max_val], 
                    mode='lines', 
                    name='Línea 1:1',
                    line=dict(color='#DC3545', dash='dash') # Rojo
                ))

                fig_scatter.update_layout(
                    title=f'Correlación Diaria: PM Estándar vs. {model_name} en {code}',
                    xaxis_title=VAR_MAP.get('ET0_calc', 'ET₀ PM (mm/día)'),
                    yaxis_title=VAR_MAP.get(model_y_column, model_y_column),
                    font=FONT_STYLE,
                    hovermode="closest",
                    template='seaborn'
                )
                
                graph_scatter = dcc.Graph(figure=fig_scatter)
            else:
                graph_scatter = html.P(f'Datos insuficientes para el modelo {model_name}.', className="alert alert-warning")

            
            # 5. GRÁFICO: Diferencias vs Variable X (Dispersión)
            diff_col = 'diff_' + model_y_column.replace('ET0_', '')
            if diff_col in df_filtered.columns and var_x in df_filtered.columns:
                
                fig_diff_var = px.scatter(
                    df_filtered,
                    x=var_x,
                    y=diff_col,
                    title=f'Error (PM - {model_name}) vs. {var_x_name} en {code}',
                    labels={var_x: var_x_name, diff_col: 'Error (PM - Modelo) (mm/día)'},
                    opacity=0.5,
                    template='seaborn'
                )
                # Añadir línea de Bias 0
                fig_diff_var.add_hline(y=0, line_dash="dash", line_color="#000", annotation_text="Bias = 0")
                fig_diff_var.update_layout(
                    font=FONT_STYLE, 
                    hovermode="closest"
                )
                
                graph_diff_var = dcc.Graph(figure=fig_diff_var)
            else:
                graph_diff_var = html.P(f'Datos de diferencia o variable ({var_x_name}) insuficientes.', className="alert alert-warning")

            
            # 6. GRÁFICO: Error Medio por Mes (Bias Mensual)
            df_diff_month_mean = df_filtered.groupby('Mes').agg({
                'diff_harg_ajustado': 'mean',
                'diff_val_ajustado': 'mean',
                'diff_sun': 'mean'
            }).reset_index()

            # Melt para el gráfico de barras (muestra todos los modelos de error a la vez)
            df_diff_month_melt = df_diff_month_mean.melt(
                id_vars='Mes',
                value_vars=[col for col in ['diff_harg_ajustado', 'diff_val_ajustado', 'diff_sun'] if col in df_diff_month_mean.columns],
                var_name='Diferencia',
                value_name='Bias Medio (mm/día)'
            )
            
            # Mapeo de Mes a nombre
            meses_map = {i: pd.to_datetime(f'2024-{i}-01').strftime('%b') for i in range(1, 13)}
            df_diff_month_melt['Mes_Nombre'] = df_diff_month_melt['Mes'].map(meses_map)
            
            fig_diff_month = px.bar(
                df_diff_month_melt,
                x='Mes_Nombre',
                y='Bias Medio (mm/día)',
                color='Diferencia',
                barmode='group',
                title=f'Error Medio Mensual (Bias: PM - Modelo) en {code}',
                labels={'Mes_Nombre': 'Mes', 'Diferencia': 'Modelo'},
                color_discrete_map={
                    'diff_harg_ajustado': '#28A745', 
                    'diff_val_ajustado': '#FFC107',
                    'diff_sun': '#DC3545'
                },
                template='seaborn'
            )
            fig_diff_month.add_hline(y=0, line_dash="dash", line_color="#000")
            fig_diff_month.update_layout(font=FONT_STYLE, hovermode="x unified")
            fig_diff_month.for_each_trace(lambda t: t.update(name=t.name.replace('diff_', '').replace('_ajustado', ' Ajustado')))

            graph_diff_month = dbc.Card([
                dbc.CardHeader(html.H3('Error Medio Mensual (Bias)', style={'color': '#2c3e50', 'fontSize': '1.25rem'})),
                dbc.CardBody(dcc.Graph(figure=fig_diff_month))
            ], className="mb-4 shadow-sm border-info")


            # 7. Construcción del contenido final
            content = html.Div([
                # Fila de gráfico de serie temporal
                dbc.Row(dbc.Col(graph_time, className="mb-4")),
                
                # Fila de gráficos de análisis (Scatter y Diferencias vs Variable)
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader(html.H3(f'Correlación PM vs {model_name}', style={'color': '#2c3e50', 'fontSize': '1.25rem'})),
                        dbc.CardBody(graph_scatter)
                    ], className="h-100 shadow-sm border-success"), md=6),
                    dbc.Col(dbc.Card([
                        dbc.CardHeader(html.H3(f'Error vs. {var_x_name} (Dispersión)', style={'color': '#2c3e50', 'fontSize': '1.25rem'})),
                        dbc.CardBody(graph_diff_var)
                    ], className="h-100 shadow-sm border-success"), md=6),
                ], className="mb-4"),
                
                # Fila de tabla de errores y Bias Mensual
                dbc.Row([
                    dbc.Col(error_table, md=5),
                    dbc.Col(graph_diff_month, md=7),
                ], className="mb-4"),
            ])
            
            return content, error_msg
        
        except Exception as e:
            error_msg = f"🚨 Error crítico al actualizar el dashboard: {str(e)}"
            print(error_msg)
            # Retorna una vista de error amigable en la UI
            return html.Div([
                html.H3("¡Error Inesperado en el Dashboard!", style={'color': '#e74c3c'}),
                html.P(f"El procesamiento falló. Detalle: {str(e)}"),
                html.P("Por favor, revisa la consola para más detalles."),
            ], className="alert alert-danger"), error_msg

    elif not AVAILABLE_ESTACIONES:
        error_msg = '🚨 No se encontraron archivos de datos válidos. Asegúrate de que la carpeta "datos_siar_baleares" existe y contiene los CSVs.'
        return html.Div(html.P(error_msg, className="alert alert-danger")), error_msg

    return html.Div(), error_msg