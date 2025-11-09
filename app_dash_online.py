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

# Opcional: para limpiar logs de warnings de pandas/numpy, deshabilitado en producción si hay problemas
# warnings.filterwarnings('ignore') 

# Inicializar la app con tema Bootstrap para diseño moderno
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# VARIABLE CRÍTICA para que Render/Gunicorn pueda servir la aplicación
server = app.server 

# =========================================================================
# 1. CARGA GLOBAL DE DATOS (Optimización y Preparación para el Servidor)
# =========================================================================
DATA_PATH = 'datos_siar_baleares'
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
            print(f"⚠️ Archivo no encontrado para {code}: {filepath}")
            return None
        
        df = pd.read_csv(filepath)
        df['Estacion'] = code
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        df['Mes'] = df['Fecha'].dt.month
        
        # Filtrar columnas necesarias y renombrar para claridad
        cols_to_keep = [
            'Fecha', 'Estacion', 'Mes', 'TempMedia', 'Radiacion', 'ET0_calc', 
            'ET0_harg_ajustado', 'ET0_val_ajustado', 'ET0_sun'
        ]
        
        # Asegurar que solo se mantienen las columnas existentes
        df = df[[col for col in cols_to_keep if col in df.columns]]

        # Calcular diferencias vs. ET0_calc (referencia)
        df['diff_harg_ajustado'] = df['ET0_calc'] - df['ET0_harg_ajustado']
        df['diff_val_ajustado'] = df['ET0_calc'] - df['ET0_val_ajustado']
        df['diff_sun'] = df['ET0_calc'] - df['ET0_sun']

        # Añadir columna de Año
        df['Año'] = df['Fecha'].dt.year

        return df
    except Exception as e:
        print(f"🚨 Error leyendo o procesando {filepath}: {e}")
        return None

def load_data_globally():
    """Carga todos los datos usando multithreading para mayor velocidad."""
    global DF_ALL, AVAILABLE_ESTACIONES, AVAILABLE_YEARS
    
    print(f"Iniciando carga global de archivos CSV desde {DATA_PATH}...")
    
    # Usamos ThreadPoolExecutor para paralelizar la lectura de archivos
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(load_station_data, ESTACIONES_COD))
        
    loaded_dfs = [res for res in results if res is not None]
    
    if loaded_dfs:
        DF_ALL = pd.concat(loaded_dfs, ignore_index=True)
        # Limpiar NaNs en las columnas de ET0 para asegurar cálculos fiables
        DF_ALL.dropna(subset=['ET0_calc', 'ET0_val_ajustado', 'ET0_harg_ajustado'], inplace=True)
        
        AVAILABLE_ESTACIONES = sorted(DF_ALL['Estacion'].unique().tolist())
        AVAILABLE_YEARS = sorted(DF_ALL['Año'].unique().tolist())

        print(f"✅ Carga global completa. Estaciones disponibles: {AVAILABLE_ESTACIONES}")
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

def calculate_errors_for_station(df_station, code):
    """Calcula las métricas de error (MAE, R2, etc.) por estación y por modelo."""
    if df_station.empty or len(df_station) < 5:
        return None

    models = {
        'PM Cielo Claro': 'ET0_sun',
        'Hargreaves Ajustado': 'ET0_harg_ajustado',
        'Valiantzas Ajustado': 'ET0_val_ajustado'
    }

    errors = []
    y_true = df_station['ET0_calc']
    
    for model_name, col_name in models.items():
        y_pred = df_station[col_name]
        
        # Filtrar nulos si existen
        mask = y_true.notna() & y_pred.notna()
        if not mask.any(): continue
            
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        # Métrica: Mean Absolute Error (MAE)
        mae = np.mean(np.abs(y_true_clean - y_pred_clean))
        
        # Métrica: R² (Coeficiente de Determinación)
        try:
            r2 = r2_score(y_true_clean, y_pred_clean)
        except Exception:
            r2 = np.nan
        
        # Métrica: Root Mean Squared Error (RMSE)
        rmse = np.sqrt(np.mean((y_true_clean - y_pred_clean)**2))
        
        # Métrica: Error Medio (Bias)
        bias = np.mean(y_pred_clean - y_true_clean)

        errors.append({
            'Estación': code,
            'Modelo': model_name,
            'N': len(y_true_clean),
            'Bias (mm/día)': f'{bias:.3f}',
            'MAE (mm/día)': f'{mae:.3f}',
            'RMSE (mm/día)': f'{rmse:.3f}',
            'R²': f'{r2:.3f}'
        })

    return errors

def r2_score(y_true, y_pred):
    """Implementación de R² para evitar dependencia de sklearn."""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot == 0:
        return np.nan
    return 1 - (ss_res / ss_tot)


# =========================================================================
# 3. LAYOUT DEL DASHBOARD
# =========================================================================

app.layout = dbc.Container([
    # Título principal
    dbc.Row(dbc.Col(html.H1(
        "ET₀ Islas Baleares: Análisis de Modelos Empíricos",
        className="text-center my-4",
        style={'color': '#007BFF'}
    ))),

    # Fila de controles (Estación, Año, Variables)
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
        ], className="h-100"), md=4),

        # RangeSlider de Año
        dbc.Col(dbc.Card([
            dbc.CardHeader("Rango de Años"),
            dbc.CardBody(dcc.RangeSlider(
                id='year-slider',
                min=min(AVAILABLE_YEARS) if AVAILABLE_YEARS else 2000,
                max=max(AVAILABLE_YEARS) if AVAILABLE_YEARS else 2024,
                value=[min(AVAILABLE_YEARS), max(AVAILABLE_YEARS)] if AVAILABLE_YEARS else [2000, 2024],
                step=1,
                marks={year: str(year) for year in AVAILABLE_YEARS if year % 4 == 0 or year == min(AVAILABLE_YEARS) or year == max(AVAILABLE_YEARS)},
                tooltip={"placement": "bottom", "always_visible": True}
            ))
        ], className="h-100"), md=5),
        
        # Dropdown de Variable para Scatter Plot
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
        ], className="h-100"), md=3),

    ], className="mb-4"),

    # Fila de Mensajes de Error (al inicio invisible)
    dbc.Row(dbc.Col(html.Div(id='error-message', style={'color': 'red', 'fontWeight': 'bold'}, children=''))),

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
     Input('variable-dropdown', 'value')]
)
def update_dashboard(code, year_range, var_x):
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

            if df_filtered.empty:
                error_msg = f'⚠️ No hay datos para {code} en el rango {min_year}-{max_year}.'
                return html.Div(), error_msg

            # 2. CÁLCULO DE ERRORES (TABLA)
            errors_data = calculate_errors_for_station(df_filtered, code)
            
            if errors_data:
                errors_df = pd.DataFrame(errors_data)
                errors_columns = [{"name": i, "id": i} for i in errors_df.columns]
                
                error_table = dbc.Card([
                    dbc.CardHeader(html.H3('Análisis de Errores vs. ET₀ Penman-Monteith (SIAR)', style={'color': '#2c3e50', 'fontSize': '1.25rem'})),
                    dbc.CardBody(dash_table.DataTable(
                        id='errors-table',
                        data=errors_df.to_dict('records'),
                        columns=errors_columns,
                        style_cell=TABLE_STYLE,
                        style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                        style_table={'overflowX': 'auto'} # Scroll horizontal para tablas anchas
                    ))
                ], className="mb-4 shadow-sm")
            else:
                error_table = html.P('Datos insuficientes para calcular errores.', className="alert alert-warning")


            # 3. GRÁFICO: Serie Temporal de ET₀ (Media Mensual)
            df_monthly = df_filtered.groupby(['Año', 'Mes']).agg(
                {col: 'mean' for col in ['ET0_calc', 'ET0_harg_ajustado', 'ET0_val_ajustado', 'ET0_sun']}
            ).reset_index()
            df_monthly['Fecha_Mensual'] = pd.to_datetime(df_monthly[['Año', 'Mes']].assign(day=1))
            
            fig_time = px.line(
                df_monthly, 
                x='Fecha_Mensual', 
                y=['ET0_calc', 'ET0_harg_ajustado', 'ET0_val_ajustado', 'ET0_sun'],
                title=f'Serie Temporal de ET₀ (Media Mensual) en {code}',
                labels={'value': 'ET₀ (mm/día)', 'Fecha_Mensual': 'Fecha', 'variable': 'Modelo'},
                color_discrete_map={
                    'ET0_calc': '#007BFF',  # Azul
                    'ET0_harg_ajustado': '#28A745', # Verde
                    'ET0_val_ajustado': '#FFC107', # Amarillo
                    'ET0_sun': '#DC3545' # Rojo
                },
                template='seaborn'
            )
            fig_time.update_layout(legend_title_text='Modelo', font=FONT_STYLE, hovermode="x unified")
            fig_time.for_each_trace(lambda t: t.update(name = VAR_MAP.get(t.name, t.name))) # Renombrar leyenda
            
            graph_time = dcc.Graph(figure=fig_time)

            
            # 4. GRÁFICO: Diferencias vs. Variable X (Scatter)
            var_y_diff = VAR_MAP.get(var_x, var_x)
            
            # Melt para tener las diferencias en una sola columna para el scatter
            df_diff_melt = df_filtered.melt(
                id_vars=[var_x], 
                value_vars=['diff_harg_ajustado', 'diff_val_ajustado', 'diff_sun'],
                var_name='Diferencia', 
                value_name='Error (mm/día)'
            )
            
            fig_diff_var = px.scatter(
                df_diff_melt,
                x=var_x,
                y='Error (mm/día)',
                color='Diferencia',
                title=f'Error (PM - Modelo) vs. {var_y_diff} en {code}',
                labels={var_x: var_y_diff, 'Diferencia': 'Modelo'},
                color_discrete_map={
                    'diff_harg_ajustado': '#28A745', 
                    'diff_val_ajustado': '#FFC107',
                    'diff_sun': '#DC3545'
                },
                opacity=0.5,
                template='seaborn'
            )
            fig_diff_var.update_layout(
                legend_title_text='Modelo', 
                font=FONT_STYLE, 
                hovermode="closest"
            )
            fig_diff_var.for_each_trace(lambda t: t.update(name=t.name.replace('diff_', '').replace('_ajustado', ' Ajustado')))
            
            graph_diff_var = dcc.Graph(figure=fig_diff_var)

            # 5. GRÁFICO: Error Medio por Mes (Barra)
            # Calcular la media de las diferencias por mes
            df_diff_month_mean = df_filtered.groupby('Mes').agg({
                'diff_harg_ajustado': 'mean',
                'diff_val_ajustado': 'mean',
                'diff_sun': 'mean'
            }).reset_index()

            # Melt para el gráfico de barras
            df_diff_month_melt = df_diff_month_mean.melt(
                id_vars='Mes',
                value_vars=['diff_harg_ajustado', 'diff_val_ajustado', 'diff_sun'],
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
            fig_diff_month.update_layout(font=FONT_STYLE, hovermode="x unified")
            fig_diff_month.for_each_trace(lambda t: t.update(name=t.name.replace('diff_', '').replace('_ajustado', ' Ajustado')))

            graph_diff_month = dcc.Graph(figure=fig_diff_month)


            # 6. Contenido Final
            content = html.Div([
                # Fila de tabla de errores
                dbc.Row(dbc.Col(error_table, className="mb-4")),
                
                # Fila de gráfico de serie temporal
                dbc.Row(dbc.Col(dbc.Card([
                    dbc.CardHeader(html.H3('Serie Temporal de ET₀ (Media Mensual)', style={'color': '#2c3e50', 'fontSize': '1.25rem'})),
                    dbc.CardBody(graph_time)
                ], className="mb-4 shadow-sm"))),
                
                # Fila de gráfico de diferencias vs. variable
                dbc.Row(dbc.Col(dbc.Card([
                    dbc.CardHeader(html.H3(f'Diferencias vs. {var_y_diff} (Dispersión)', style={'color': '#2c3e50', 'fontSize': '1.25rem'})),
                    dbc.CardBody(graph_diff_var)
                ], className="mb-4 shadow-sm"))),
                
                # Fila de gráfico de error medio mensual
                dbc.Row(dbc.Col(dbc.Card([
                    dbc.CardHeader(html.H3('Error Medio Mensual (Bias)', style={'color': '#2c3e50', 'fontSize': '1.25rem'})),
                    dbc.CardBody(graph_diff_month)
                ], className="mb-4 shadow-sm"))),

            ])
            return content, error_msg
        
        except Exception as e:
            error_msg = f"🚨 Error crítico procesando la estación {code} o los datos: {str(e)}"
            print(error_msg)
            # Retornar un mensaje de error simple en la UI si falla algo más allá del filtrado
            return html.Div(html.P("Error interno del servidor. Consulta los logs para más detalles.")), error_msg

    elif not AVAILABLE_ESTACIONES:
        error_msg = '🚨 No se encontraron archivos de datos válidos en la carpeta "datos_siar_baleares".'
        return html.Div(error_msg), error_msg

    return html.Div(), error_msg

# Se mantiene el if __name__ == '__main__': para pruebas locales, pero Render usa la variable 'server'
# if __name__ == '__main__':
#     app.run_server(debug=True)