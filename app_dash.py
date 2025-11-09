import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import os
import re
from sklearn.metrics import r2_score, mean_absolute_error # Necesario para las métricas
import warnings
warnings.filterwarnings('ignore')

# Inicializar la app con tema Bootstrap para diseño moderno
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# =========================================================================
# 1. CONFIGURACIÓN Y CARGA GLOBAL DE DATOS
# =========================================================================
data_path = 'datos_siar_baleares'
global_data_store = {}
estaciones_disponibles = [] # Lista que se llenará dinámicamente

font_style = {'family': 'Arial, sans-serif', 'size': 14, 'color': '#333333'}
header_style = {'fontWeight': 'bold', 'color': '#2c3e50', 'marginBottom': '10px'}
table_style = {'textAlign': 'left', 'fontFamily': 'Arial', 'fontSize': '14px'}

def load_data_globally():
    """
    Carga todos los archivos de datos de ET0 y metadatos, y genera la lista de estaciones.
    """
    global global_data_store
    global estaciones_disponibles
    
    print("\nIniciando carga global de archivos CSV...")
    
    # 1. Detectar archivos de datos (IBXX_et0_variants_ajustado.csv)
    all_files = os.listdir(data_path)
    csv_files = [f for f in all_files if re.match(r'IB\d{2}_et0_variants_ajustado\.csv$', f)]
    global_data_store = {}
    
    for filename in csv_files:
        code = re.search(r'(IB\d{2})', filename).group(1)
        filepath = os.path.join(data_path, filename)
        
        try:
            df = pd.read_csv(filepath, sep=',')
            df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
            df.dropna(subset=['Fecha'], inplace=True)
            df.set_index('Fecha', inplace=True)
            
            # Asegurar que las columnas clave sean numéricas para evitar errores en cálculos
            numeric_cols = ['ET0_calc', 'ET0_harg', 'ET0_val', 'ET0_harg_ajustado', 'ET0_val_ajustado', 'TempMedia', 'TempMax', 'TempMin']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            global_data_store[code] = df
            print(f"Cargada estación {code}: {len(df)} filas.")
            
        except Exception as e:
            print(f"ERROR cargando o procesando {filename}: {e}")

    # 2. Generar lista de estaciones disponibles
    estaciones_disponibles = sorted(global_data_store.keys())

    # 3. Cargar metadatos desde metadata_estaciones_baleares.csv
    try:
        metadata_path = os.path.join(data_path, 'metadata_estaciones_baleares.csv')
        metadata_df = pd.read_csv(metadata_path, sep=',')
        metadata_df.set_index('Código estación', inplace=True)
        global_data_store['METADATA'] = metadata_df
        print(f"Cargados metadatos de {len(metadata_df)} estaciones.")
    except FileNotFoundError:
        print(f"Advertencia: Archivo {metadata_path} no encontrado. La tabla de información de estación estará vacía.")
    except Exception as e:
        print(f"ERROR cargando metadatos: {e}")

    if not estaciones_disponibles:
        print("ADVERTENCIA CRÍTICA: No se cargó ninguna estación de datos.")

# Cargar los datos al inicio
load_data_globally()

# =========================================================================
# 2. LAYOUT (DISEÑO) DE LA APLICACIÓN
# =========================================================================

app.layout = dbc.Container([
    html.H1("Dashboard de Evapotranspiración (ET₀) SIAR - Baleares", 
            className="my-4 text-center text-primary", 
            style={'fontWeight': '900'}),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("Seleccionar Estación", style={'color': '#2c3e50'}),
                dcc.Dropdown(
                    id='estacion-dropdown',
                    options=[{'label': f'Estación {code}', 'value': code} for code in estaciones_disponibles],
                    value=estaciones_disponibles[0] if estaciones_disponibles else None,
                    clearable=False,
                    className="mb-3",
                    style=font_style,
                ),
            ], className="p-3 bg-light rounded shadow-sm"),
        ], width=12),
    ], className="mb-4"),

    # Contenedor para el contenido principal (gráficos, tablas, etc.)
    dbc.Row(id='content-container'),
    
    # Contenedor para mensajes de error
    dbc.Row(dbc.Col(html.Div(id='error-message', className="text-danger fw-bold my-3 text-center"))),

], fluid=True, style={'maxWidth': '1400px', 'backgroundColor': '#f9f9f9', 'padding': '20px'})


# =========================================================================
# 3. CALLBACKS (LÓGICA DE INTERACTIVIDAD)
# =========================================================================

# Diccionario para mapear las columnas a nombres amigables para la leyenda y las métricas
MODEL_COLUMNS = {
    'ET0_calc': 'Penman-Monteith (SIAR Ref.)',
    'ET0_harg': 'Hargreaves (Estándar)',
    'ET0_val': 'Valiantzas (Estándar)',
    'ET0_harg_ajustado': 'Hargreaves (Ajustado)',
    'ET0_val_ajustado': 'Valiantzas (Ajustado)',
    'ET0_sun': 'Penman-Monteith (Cielo Claro)' # Suponiendo esta columna si existe
}


@app.callback(
    [Output('content-container', 'children'),
     Output('error-message', 'children')],
    [Input('estacion-dropdown', 'value')]
)
def update_dashboard(code):
    error_msg = ""
    
    if not code or code not in global_data_store:
        error_msg = f"🚨 ERROR: Estación no seleccionada o datos no cargados para '{code}'."
        return html.Div([html.P(error_msg)], className="text-center p-5"), error_msg

    try:
        df = global_data_store[code].copy()
        
        # ----------------------------------------------------
        # NUEVA SECCIÓN 1: TABLA DE INFORMACIÓN Y ESTADÍSTICAS
        # ----------------------------------------------------
        info_data = []
        
        # Metadatos de la Estación
        if 'METADATA' in global_data_store and code in global_data_store['METADATA'].index:
            meta = global_data_store['METADATA'].loc[code]
            info_data.extend([
                {'Propiedad': 'Nombre Estación', 'Valor': meta.get('Nombre estación', 'N/D')},
                {'Propiedad': 'Término Municipal', 'Valor': meta.get('Termino', 'N/D')},
                {'Propiedad': 'Altitud (m)', 'Valor': f"{meta.get('Altitud', 'N/D'):.0f}" if pd.notna(meta.get('Altitud')) else 'N/D'},
                {'Propiedad': 'Latitud', 'Valor': f"{meta.get('Latitud', 'N/D')}"},
                {'Propiedad': 'Longitud', 'Valor': f"{meta.get('Longitud', 'N/D')}"},
            ])

        # Estadísticas de los Datos
        start_date = df.index.min().strftime('%Y-%m-%d') if not df.empty and df.index.min() is not pd.NaT else 'N/D'
        end_date = df.index.max().strftime('%Y-%m-%d') if not df.empty and df.index.max() is not pd.NaT else 'N/D'

        info_data.extend([
            {'Propiedad': 'Rango de Datos', 'Valor': f"{start_date} a {end_date}"},
            {'Propiedad': 'Días de Registro', 'Valor': f"{len(df)}"},
            {'Propiedad': 'Media Temp. Media (°C)', 'Valor': f"{df['TempMedia'].mean():.2f}" if 'TempMedia' in df.columns else 'N/D'},
            {'Propiedad': 'Media Temp. Máxima (°C)', 'Valor': f"{df['TempMax'].mean():.2f}" if 'TempMax' in df.columns else 'N/D'},
            {'Propiedad': 'Media Temp. Mínima (°C)', 'Valor': f"{df['TempMin'].mean():.2f}" if 'TempMin' in df.columns else 'N/D'},
            {'Propiedad': 'Media ET₀ PM (mm/día)', 'Valor': f"{df['ET0_calc'].mean():.2f}" if 'ET0_calc' in df.columns else 'N/D'},
        ])

        info_df = pd.DataFrame(info_data)
        info_columns = [{"name": i, "id": i} for i in info_df.columns]

        # ----------------------------------------------------
        # NUEVA SECCIÓN 2: CÁLCULO DE ERRORES CONSOLIDADOS
        # ----------------------------------------------------
        
        # Modelos a evaluar vs. ET0_calc (PM SIAR)
        target_col = 'ET0_calc'
        
        # Filtramos solo los modelos que existen en el DataFrame y que no son el de referencia
        model_cols = {col: name for col, name in MODEL_COLUMNS.items() if col in df.columns and col != target_col}
        
        errors_data = []
        if target_col in df.columns:
            y_true = df[target_col].dropna()

            for col_name, model_name in model_cols.items():
                y_pred = df[col_name].loc[y_true.index]

                # Aseguramos que solo comparamos las mismas fechas
                common_index = y_true.index.intersection(y_pred.index)
                y_true_final = y_true.loc[common_index]
                y_pred_final = y_pred.loc[common_index]
                
                if len(common_index) > 0:
                    mae = mean_absolute_error(y_true_final, y_pred_final)
                    # El R2_score estándar
                    r2 = r2_score(y_true_final, y_pred_final)
                    
                    # Calcular Bias (Diferencia Media)
                    bias = np.mean(y_pred_final - y_true_final) # Positivo: sobreestima, Negativo: subestima
                    
                    errors_data.append({
                        'Modelo': model_name,
                        'MAE (mm/día)': f"{mae:.3f}",
                        'R²': f"{r2:.4f}",
                        'Bias (mm/día)': f"{bias:.3f}"
                    })

        errors_df = pd.DataFrame(errors_data)
        errors_columns = [{"name": i, "id": i} for i in errors_df.columns]

        # ----------------------------------------------------
        # SECCIÓN 3: GRÁFICO DE SERIE TEMPORAL COMPLETO
        # ----------------------------------------------------
        
        # Seleccionamos TODAS las columnas de modelos disponibles
        cols_for_plot = [col for col in MODEL_COLUMNS.keys() if col in df.columns]
        df_time_series = df[cols_for_plot].copy()
        
        # Renombrar columnas para la leyenda
        df_time_series.rename(columns=MODEL_COLUMNS, inplace=True)
        
        # Resample a día para asegurar una línea continua y hacer melt
        df_plot_melt = df_time_series.resample('D').mean().dropna(how='all').reset_index().melt(
            id_vars='Fecha', 
            var_name='Modelo', 
            value_name='ET₀ (mm/día)'
        )
        
        fig_time = px.line(
            df_plot_melt,
            x='Fecha', 
            y='ET₀ (mm/día)', 
            color='Modelo',
            title=f"Serie Temporal Diaria de ET₀ (Comparación de Modelos) - Estación {code}",
            labels={'ET₀ (mm/día)': 'ET₀ (mm/día)', 'Modelo': 'Modelo'},
            template='plotly_white'
        )
        fig_time.update_traces(opacity=0.8, mode='lines')
        fig_time.update_layout(height=600, font=font_style, legend_title_text='Modelos de ET₀')


        # ----------------------------------------------------
        # 4. GENERACIÓN DE CONTENIDO FINAL
        # ----------------------------------------------------

        content = [
            html.H2(f'Análisis de Evapotranspiración para la Estación {code}', 
                    className="mt-3 mb-4 text-center text-secondary"),

            # Primera Fila: Info de la Estación
            dbc.Row([
                dbc.Col([
                    html.H3('1. Información y Estadísticas Generales de la Estación', style=header_style),
                    dash_table.DataTable(
                        id='info-table',
                        data=info_df.to_dict('records'),
                        columns=info_columns,
                        style_cell={**table_style, 'minWidth': '150px', 'width': '50%', 'maxWidth': '50%', 'padding': '10px'},
                        style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                        style_table={'marginBottom': '20px', 'borderRadius': '8px', 'overflow': 'hidden'},
                    ),
                ], width=12),
            ], className="mb-4"),


            # Segunda Fila: Tabla de Errores Consolidados
            dbc.Row([
                dbc.Col([
                    html.H3('2. Resumen de Errores (Métricas vs. Penman-Monteith SIAR)', style=header_style),
                    dash_table.DataTable(
                        id='errors-table',
                        data=errors_df.to_dict('records'),
                        columns=errors_columns,
                        style_cell={**table_style, 'minWidth': '120px', 'width': '25%', 'maxWidth': '25%'},
                        style_header={'fontWeight': 'bold', 'backgroundColor': '#e9ecef'},
                        style_table={'marginBottom': '20px', 'borderRadius': '8px', 'overflow': 'hidden'},
                    ),
                    dbc.Alert(
                        "**MAE**: Error Absoluto Medio (cuánto se desvía en promedio). **R²**: Coeficiente de Determinación (cercano a 1 es mejor). **Bias**: Diferencia Media (positivo = sobreestima, negativo = subestima).", 
                        color="secondary", 
                        className="mt-2"
                    )
                ], width=12),
            ], className="mb-4"),
            
            # Tercera Fila: Gráfico de Serie Temporal (TODOS los modelos)
            dbc.Row([
                dbc.Col([
                    html.H3('3. Serie Temporal Diaria de ET₀ (Todos los Modelos)', style=header_style),
                    dcc.Graph(figure=fig_time),
                ], width=12)
            ]),
        ]
        return content, ""
    
    except Exception as e:
        error_msg = f"🚨 Error crítico al generar el dashboard para {code}. Detalles: {str(e)}"
        print(error_msg)
        return html.Div([html.P(error_msg, className="text-danger p-5 border rounded bg-light")]), error_msg

# Nota: La ejecución se mantiene igual
if __name__ == '__main__':
    print("\n--- Ejecutando Dash App V2 ---")
    print(f"Estaciones disponibles: {estaciones_disponibles}")
    print("Accede a http://127.0.0.1:8050/ en tu navegador.")
    app.run_server(debug=True, host='0.0.0.0', port=8050)