import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime

# Ignorar warnings de pandas/numpy para limpiar logs
warnings.filterwarnings('ignore')

# Inicializar la app con tema Bootstrap para diseño moderno
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Esencial para Render/Gunicorn

# =========================================================================
# 1. CARGA GLOBAL DE DATOS
# =========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
data_path_priority = os.path.join(SCRIPT_DIR, 'datos_siar_baleares')
data_path_fallback = SCRIPT_DIR

# Estaciones y Modelos
estaciones = ['IB01', 'IB02', 'IB03', 'IB04', 'IB05'] # Extensible
modelos = {
    'PM Estándar': 'diff_pm_estandar',
    'PM Cielo Claro': 'diff_pm_sun',
    'Hargreaves': 'diff_harg',
    'Valiantzas': 'diff_val',
    'Hargreaves Ajustado': 'diff_harg_ajustado',
    'Valiantzas Ajustado': 'diff_val_ajustado'
}
corr_variables = ['TempMedia', 'Radiacion', 'HumedadMedia', 'VelViento']

# Estilos
font_style = {'family': 'Arial, sans-serif', 'size': 14, 'color': '#333333'}
header_style = {'fontWeight': 'bold', 'color': '#2c3e50', 'marginBottom': '10px'}
table_style = {'textAlign': 'left', 'fontFamily': 'Arial', 'fontSize': '14px'}

def load_data_globally():
    """Carga y concatena todos los archivos de datos de ET0."""
    df_all = pd.DataFrame()
    found_estaciones = []
    
    print("Iniciando carga global de archivos CSV...")
    
    # Intentar cargar desde la ruta prioritaria (si se está ejecutando en un entorno estructurado)
    # y luego desde el directorio del script (fallback)
    data_paths = [data_path_priority, data_path_fallback]
    
    for path in data_paths:
        if os.path.exists(path):
            print(f"Buscando archivos en: {path}")
            for est in estaciones:
                # El archivo de datos del usuario es 'IB02_et0_variants_ajustado.csv'
                filename = f'{est}_et0_variants_ajustado.csv'
                filepath = os.path.join(path, filename)

                # Si el archivo existe, cargarlo
                if os.path.exists(filepath):
                    try:
                        df = pd.read_csv(filepath)
                        
                        # CRUCIAL: Convertir la columna de fecha a datetime ANTES de concatenar
                        if 'Fecha' in df.columns:
                            df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
                            df = df.dropna(subset=['Fecha'])
                        
                        df_all = pd.concat([df_all, df], ignore_index=True)
                        found_estaciones.append(est)
                        print(f"Cargado exitosamente: {filename}")
                    except Exception as e:
                        print(f"Error al cargar {filename}: {e}")
            
            if not df_all.empty:
                break # Romper si ya se encontró data

    if df_all.empty:
        print("🚨 CRÍTICO: No se encontraron archivos de datos ET0 en las rutas especificadas.")
        # Crear un DataFrame vacío con las columnas esperadas como fallback para evitar errores
        empty_cols = ['Fecha', 'Estacion', 'TempMedia', 'Radiacion'] + list(modelos.values())
        df_all = pd.DataFrame(columns=empty_cols)
    
    # Cargar los errores globales (si existe el archivo)
    try:
        errors_path = os.path.join(SCRIPT_DIR, 'analisis_errores_global.csv')
        df_global_errors = pd.read_csv(errors_path)
    except:
        df_global_errors = pd.DataFrame(columns=['Estacion', 'Modelo', 'MSE', 'RRMSE', 'MAE', 'R2', 'AARE'])


    print(f"Carga global finalizada. Estaciones con datos: {list(set(found_estaciones))}")
    return df_all, found_estaciones, df_global_errors

# Cargar los datos una sola vez al inicio de la aplicación
DF_ALL, FOUND_ESTACIONES, DF_GLOBAL_ERRORS = load_data_globally()

# Obtener rango de años dinámico para los sliders
if not DF_ALL.empty:
    min_year = DF_ALL['Fecha'].dt.year.min()
    max_year = DF_ALL['Fecha'].dt.year.max()
else:
    min_year, max_year = 2004, 2024 # Valores por defecto

years_range = range(min_year, max_year + 1)
marks_years = {year: str(year) for year in years_range if year % 4 == 0 or year == min_year or year == max_year}

# =========================================================================
# 2. LAYOUT DEL DASHBOARD
# =========================================================================
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("ET₀ Islas Baleares: Análisis de Modelos Empíricos", 
                       className="text-center my-4 text-primary", 
                       style=header_style), 
                width=12)
    ]),
    
    # Controles de Usuario
    dbc.Row([
        # Columna 1: Selección de Estación
        dbc.Col([
            html.Label("Seleccionar Estación", style=header_style),
            dcc.Dropdown(
                id='estacion-dropdown',
                options=[{'label': est, 'value': est} for est in FOUND_ESTACIONES],
                value=FOUND_ESTACIONES[0] if FOUND_ESTACIONES else None,
                clearable=False,
                className="mb-3"
            ),
            html.Label("Modelo a Comparar", style=header_style),
            dcc.Dropdown(
                id='modelo-dropdown',
                options=[{'label': k, 'value': k} for k in modelos.keys()],
                value='Valiantzas Ajustado', # Valor por defecto
                clearable=False,
                className="mb-3"
            ),
            html.Label("Variable para Correlación", style=header_style),
            dcc.Dropdown(
                id='corr-dropdown',
                options=[{'label': v, 'value': v} for v in corr_variables],
                value='TempMedia',
                clearable=False,
                className="mb-3"
            ),
        ], md=3),
        
        # Columna 2: Rango de Años
        dbc.Col([
            html.Label("Rango de Años", style=header_style),
            dcc.RangeSlider(
                id='year-slider',
                min=min_year,
                max=max_year,
                step=1,
                value=[min_year, max_year],
                marks=marks_years,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], md=9, className="align-self-center"),
    ], className="bg-light p-4 rounded shadow-sm mb-4"),
    
    # Sección de Mensajes/Errores
    dbc.Row([
        dbc.Col(html.Div(id='error-output', className="alert alert-danger d-none"), width=12)
    ]),

    # Contenido Dinámico del Dashboard
    dbc.Row([
        dbc.Col(html.Div(id='dashboard-content'), width=12)
    ]),

], fluid=True)


# =========================================================================
# 3. CALLBACK PRINCIPAL
# =========================================================================
@app.callback(
    [Output('dashboard-content', 'children'),
     Output('error-output', 'children'),
     Output('error-output', 'className')],
    [Input('estacion-dropdown', 'value'),
     Input('year-slider', 'value'),
     Input('modelo-dropdown', 'value'),
     Input('corr-dropdown', 'value')]
)
def update_dashboard(code, year_range, model, corr_var):
    error_msg = ""
    error_class = "alert alert-danger d-none"
    
    if not code or not year_range or not model or DF_ALL.empty:
        if DF_ALL.empty:
            error_msg = "🚨 Error de Carga: No hay datos disponibles. Revisa si los CSV están en 'datos_siar_baleares'."
            error_class = "alert alert-danger"
        else:
            error_msg = "🚨 Error de Selección: Por favor, selecciona la estación, el rango de años y el modelo."
            error_class = "alert alert-danger"
        return html.Div(), error_msg, error_class

    try:
        # 1. Filtrado y Preparación de Datos
        start_year, end_year = year_range
        df_filtered = DF_ALL[
            (DF_ALL['Estacion'] == code) & 
            (DF_ALL['Fecha'].dt.year >= start_year) & 
            (DF_ALL['Fecha'].dt.year <= end_year)
        ].copy() # Usar .copy() para evitar SettingWithCopyWarning
        
        # Asegurar que la columna de referencia (EtPMon) y la variable de correlación (corr_var) existen
        required_cols = ['EtPMon', corr_var]
        diff_column = modelos.get(model)
        if diff_column not in df_filtered.columns:
            error_msg = f"🚨 Columna de diferencia '{diff_column}' (Modelo '{model}') no encontrada en los datos."
            raise ValueError(error_msg)
            
        required_cols.append(diff_column)
        
        # Eliminar filas con NaN en las columnas necesarias
        df_filtered = df_filtered.dropna(subset=required_cols)

        if df_filtered.empty:
            error_msg = f"🚨 No hay datos disponibles para la estación {code} en el rango {start_year}-{end_year} después de la limpieza."
            error_class = "alert alert-warning"
            return html.Div(), error_msg, error_class

        
        # 2. Cálculo de Errores Específicos para el Modelo Seleccionado
        # Filtrar la tabla de errores global por estación
        errors_df_station = DF_GLOBAL_ERRORS[DF_GLOBAL_ERRORS['Estacion'] == code].copy()
        errors_df_station = errors_df_station.rename(columns={'Modelo': 'Modelo Empírico'})
        
        # Resaltar el modelo seleccionado
        highlight_style = [{'if': {'filter_query': f'{{Modelo Empírico}} = "{model}"'}, 
                            'backgroundColor': '#e8f5e9', 'fontWeight': 'bold'}]

        # 3. GENERACIÓN DE GRÁFICOS
        
        # 3.1 Serie Temporal de ET₀ (EtPMon vs Modelo Seleccionado)
        fig_time = px.line(df_filtered, x='Fecha', y=['EtPMon', df_filtered[diff_column].name.replace('diff_', 'ET0_')], 
                           title=f'Serie Temporal: ET₀ de Referencia vs. {model} ({code})',
                           labels={'value': 'ET₀ (mm/día)', 'variable': 'Modelo'},
                           template='plotly_white')
        fig_time.update_layout(legend_title_text='Modelo', font=font_style, hovermode="x unified")
        
        # 3.2 Gráfico de Correlación (Diferencia vs Variable Seleccionada)
        fig_scatter = px.scatter(df_filtered, x=corr_var, y=diff_column, 
                                 title=f'Diferencia (EtPMon - {model}) vs. {corr_var}',
                                 labels={'y': 'Diferencia ET₀ (mm/día)', 'x': corr_var},
                                 template='plotly_white', 
                                 opacity=0.6)
        fig_scatter.update_layout(font=font_style, hovermode="closest")
        
        # 3.3 CRUCIAL: Agregación de Diferencias Mensuales (Corrección del Error)
        df_monthly_avg = df_filtered.copy()
        df_monthly_avg['Month_Num'] = df_monthly_avg['Fecha'].dt.month
        # Usamos el nombre corto del mes para el gráfico
        df_monthly_avg['Month_Name'] = df_monthly_avg['Fecha'].dt.strftime('%b') 

        # Agrupar por número y nombre del mes para obtener el ciclo anual de la diferencia media
        # El .name.replace('diff_', '') es para obtener el nombre "ET0_harg" o similar
        et0_model_column = diff_column.replace('diff_', 'ET0_')
        
        # Calculamos la diferencia real para el promedio mensual
        df_monthly_avg['Diff_Value'] = df_monthly_avg['EtPMon'] - df_monthly_avg[et0_model_column]
        
        df_monthly_mean = df_monthly_avg.groupby(['Month_Num', 'Month_Name']).agg(
            Mean_Diff=('Diff_Value', 'mean')
        ).reset_index().sort_values('Month_Num')
        
        # Gráfico de barras de la diferencia media mensual
        fig_diff_month = px.bar(df_monthly_mean, x='Month_Name', y='Mean_Diff', 
                                title=f'Diferencia Media Mensual (EtPMon - {model})',
                                labels={'Month_Name': 'Mes', 'Mean_Diff': 'Diferencia Media (mm/día)'},
                                template='plotly_white',
                                color=df_monthly_mean['Mean_Diff'], 
                                color_continuous_scale=px.colors.diverging.RdBu_r)
        fig_diff_month.update_layout(font=font_style)

        
        # 4. COMPONENTES HTML
        
        # Tablas de Errores
        errors_columns = [{'name': col, 'id': col} for col in errors_df_station.columns]

        table_current_errors = html.Div([
            html.H3(f'Métricas de Error por Modelo (Estación {code}, Rango {start_year}-{end_year})', style=header_style),
            dash_table.DataTable(
                id='errors-table-station',
                data=errors_df_station.to_dict('records'),
                columns=errors_columns,
                style_cell=table_style,
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                style_table={'marginBottom': '20px'},
                style_data_conditional=highlight_style # Resaltar el modelo seleccionado
            )
        ])
        
        table_global_errors = html.Div([
            html.H3('Métricas de Error Global (Media de todas las estaciones)', style=header_style),
            dash_table.DataTable(
                id='errors-table-global',
                data=DF_GLOBAL_ERRORS.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in DF_GLOBAL_ERRORS.columns],
                style_cell=table_style,
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                style_table={'marginBottom': '20px'},
            )
        ])
        
        # Ensamblaje del contenido del dashboard
        dashboard_title = dbc.Alert(f"Estación: {code} | Modelo de Comparación: {model} | Periodo: {start_year} - {end_year}", 
                                    color="success", className="mb-4 shadow-lg border-success")

        content = [dashboard_title]
        
        content.append(html.H3('Serie Temporal de ET₀', style=header_style))
        content.append(dcc.Graph(figure=fig_time, config={'displayModeBar': False}))
        
        content.append(html.H3(f'Análisis de Correlación y Diferencias ({model})', style=header_style))
        content.append(dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_scatter, config={'displayModeBar': False}), md=6),
            dbc.Col(dcc.Graph(figure=fig_diff_month, config={'displayModeBar': False}), md=6),
        ], className="mb-4"))

        content.append(table_current_errors)
        content.append(table_global_errors)
        
        # No hay error, así que la clase de error se oculta
        return content, error_msg, "alert alert-danger d-none"
    
    except Exception as e:
        error_msg = f"🚨 Error crítico al actualizar el dashboard: {str(e)}"
        print(f"ERROR EN CONSOLA: {error_msg}")
        error_class = "alert alert-danger"
        # Retorna una vista de error amigable en la app
        return html.Div([
            html.H3("¡Error Inesperado en el Dashboard!", style={'color': '#e74c3c'}),
            html.P(f"El procesamiento falló. Detalle: {str(e)}"),
            html.P("Por favor, revisa la consola para más detalles."),
        ], className="alert alert-danger"), error_msg, error_class

if __name__ == '__main__':
    # La versión de Dash instalada usa app.run_server()
    # En un entorno como Render/Gunicorn, se usa el objeto 'server' directamente.
    print("Dash app starting...")
    # Solo ejecutar si se corre localmente
    # app.run_server(debug=True) # Descomentar para desarrollo local
    
# NOTA: En el entorno de Canvas, solo se debe generar el archivo.
# El error 'to assemble mappings requires...' está corregido en la sección 3.3.