import dash
from dash import dcc, html, Input, Output, dash_table, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore') # Opcional: para limpiar logs de warnings

# --- Configuración Inicial ---
# Inicializar la app con tema Bootstrap para diseño moderno
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Esencial para Render/Gunicorn

# --- Constantes y Definiciones de Modelos ---
data_path = 'datos_siar_baleares'
# Extensible: el script ahora encontrará las estaciones en la carpeta
# estaciones_base = ['IB01', 'IB02', 'IB03', 'IB04', 'IB05'] 

# Definiciones de modelos para un manejo más limpio
# El script buscará estos prefijos y los tratará dinámicamente
MODEL_DEFINITIONS = {
    'EtPMon': {'nombre': 'SIAR (EtPMon)', 'tipo': 'Referencia'},
    'ET0_calc': {'nombre': 'PM Estándar', 'tipo': 'Empírico'},
    'ET0_sun': {'nombre': 'PM Cielo Claro', 'tipo': 'Empírico'},
    'ET0_harg': {'nombre': 'Hargreaves', 'tipo': 'Empírico'},
    'ET0_val': {'nombre': 'Valiantzas', 'tipo': 'Empírico'},
    'ET0_harg_aj': {'nombre': 'Hargreaves Ajustado', 'tipo': 'Empírico'},
    'ET0_val_aj': {'nombre': 'Valiantzas Ajustado', 'tipo': 'Empírico'},
    'ANN_': {'nombre': 'Modelo ANN', 'tipo': 'ANN'} # Prefijo para modelos ANN
}

# Estilos
font_style = {'family': 'Arial, sans-serif', 'size': 14, 'color': '#333333'}
header_style = {'fontWeight': 'bold', 'color': '#2c3e50', 'marginBottom': '10px'}
table_style = {'textAlign': 'left', 'fontFamily': 'Arial', 'fontSize': '14px'}

# =========================================================================
# 1. CARGA GLOBAL DE DATOS (Optimizado)
# =========================================================================

def load_data_globally():
    """Carga y concatena todos los archivos de datos de ET0."""
    df_all = pd.DataFrame()
    found_estaciones = []
    
    print("Iniciando carga global de archivos CSV...")
    if not os.path.exists(data_path):
        print(f"ADVERTENCIA: El directorio de datos '{data_path}' no existe.")
        return pd.DataFrame(), []

    # Detectar estaciones automáticamente de los archivos
    try:
        all_files = os.listdir(data_path)
        estaciones = sorted(list(set([f.split('_')[0] for f in all_files if f.endswith('_et0_variants.csv')])))
        print(f"Estaciones detectadas: {estaciones}")
    except Exception as e:
        print(f"Error detectando estaciones: {e}")
        return pd.DataFrame(), []

    for est in estaciones:
        file = os.path.join(data_path, f'{est}_et0_variants.csv')
        if os.path.exists(file):
            try:
                # Usar 'utf-8-sig' para manejar el Byte Order Mark (BOM)
                df = pd.read_csv(file, encoding='utf-8-sig', encoding_errors='replace', dtype={'Estacion': str})
                df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce', dayfirst=True) 
                df['Estacion'] = est
                df_all = pd.concat([df_all, df], ignore_index=True)
                found_estaciones.append(est)
                print(f"Cargada estación {est}: {len(df)} filas. Columnas: {list(df.columns)}")
            except Exception as e:
                print(f"Error al cargar {file}: {e}")
        else:
            print(f"Archivo no encontrado (esperado): {file}")
            
    if df_all.empty:
        print("ADVERTENCIA: df_all está vacío. Los gráficos no funcionarán.")
    
    df_all = df_all.dropna(subset=['Fecha']).sort_values(by='Fecha')
    return df_all, found_estaciones

# Carga de datos de ET0
try:
    df_all, estaciones_disponibles = load_data_globally()
except Exception as e:
    print(f"Error fatal de carga: {e}")
    df_all = pd.DataFrame()
    estaciones_disponibles = []

# Cargar metadatos de estaciones
estaciones_csv_path = os.path.join(data_path, 'estaciones_baleares.csv')
estaciones_df = pd.DataFrame()
if os.path.exists(estaciones_csv_path):
    try:
        estaciones_df = pd.read_csv(estaciones_csv_path, sep=',', encoding='utf-8-sig', quotechar='"', engine='python', encoding_errors='replace')
        estaciones_df['Codigo'] = estaciones_df['Codigo'].astype(str).str.strip().str.upper() 
        print(f"Cargados metadatos de {len(estaciones_df)} estaciones.")
    except Exception as e:
        print(f"Error al cargar metadatos de estaciones: {e}")
else:
    print(f"Advertencia: Archivo {estaciones_csv_path} no encontrado.")

# Cargar AHC
ahc_csv_path = os.path.join(data_path, 'ahc_por_estacion.csv')
ahc_df = pd.DataFrame(columns=['Estacion', 'AHC_Hargreaves', 'AHC_Valiantzas'])
if os.path.exists(ahc_csv_path):
    try:
        ahc_df = pd.read_csv(ahc_csv_path, sep=',', encoding='utf-8-sig')
        ahc_df['Estacion'] = ahc_df['Estacion'].astype(str).str.strip().str.upper() 
        print(f"Cargados coeficientes AHC de {len(ahc_df)} estaciones.")
    except Exception as e:
        print(f"Advertencia: No se pudo cargar ahc_por_estacion.csv. {e}")
else:
    print(f"Advertencia: Archivo {ahc_csv_path} no encontrado.")

# Datos TFG (hard-coded de docs_1.3_Análisis errores.md)
tfg_data = {
    'Modelo': ['Hargreaves', 'Hargreaves', 'Valiantzas', 'Valiantzas', 'Hargreaves Ajustado', 'Hargreaves Ajustado', 'Valiantzas Ajustado', 'Valiantzas Ajustado'],
    'Fuente': ['TFG', 'Python', 'TFG', 'Python', 'TFG', 'Python', 'TFG', 'Python'],
    'MSE (mm²/día²)': [0.693, 0.56, 0.381, 0.24, 0.378, 0.31, 0.250, 0.21],
    'RRMSE': [0.280, 0.24, 0.204, 0.16, 0.202, 0.18, 0.166, 0.15],
    'MAE (mm/día)': [0.672, 0.61, 0.461, 0.37, 0.468, 0.43, 0.373, 0.35],
    'R²': [0.889, 0.91, 0.924, 0.94, 0.889, 0.91, 0.924, 0.94],
    'AARE': [0.326, 0.29, 0.211, 0.16, 0.222, 0.18, 0.165, 0.14]
}
tfg_df = pd.DataFrame(tfg_data)

# =========================================================================
# 2. DEFINICIÓN DE FUNCIONES DE UTILIDAD (Fuera de los Callbacks)
# =========================================================================
def calculate_errors(obs, est):
    """Calcula las métricas de error entre la serie observada (obs) y la estimada (est)."""
    valid = (~np.isnan(obs)) & (~np.isnan(est))
    obs, est = obs[valid], est[valid]
    
    if len(obs) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan
        
    mse = np.mean((obs - est)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(obs - est))
    
    r2 = np.nan
    try:
        if np.std(obs) > 0 and np.std(est) > 0:
            r2 = np.corrcoef(obs, est)[0,1]**2
    except:
        pass 
        
    aare = np.nan
    valid_aare = valid & (obs != 0)
    obs_aare, est_aare = obs[valid_aare], est[valid_aare]
    if len(obs_aare) > 0:
        aare = np.mean(np.abs((obs_aare - est_aare) / obs_aare))
        
    rrmse_val = rmse / np.mean(obs) if np.mean(obs) != 0 else np.nan
    
    return round(mse, 3), round(rrmse_val, 3), round(mae, 3), round(r2, 3), round(aare, 3)

def get_model_name(col):
    """Obtiene el nombre bonito de una columna de modelo."""
    if col in MODEL_DEFINITIONS:
        return MODEL_DEFINITIONS[col]['nombre']
    if col.startswith('ANN_'):
        # Para columnas personalizadas como ANN_HR_10_neuronas
        return col.replace('_', ' ')
    return col

def calculate_global_errors(df):
    """Calcula errores para todos los modelos en el dataframe global."""
    if 'EtPMon' not in df.columns:
        return pd.DataFrame()
        
    obs = df['EtPMon'].values
    error_names = ['MSE (mm²/día²)', 'RRMSE', 'MAE (mm/día)', 'R²', 'AARE']
    mean_errors = {}
    
    # Identificar todas las columnas de modelo (empíricas y ANN)
    model_cols = []
    for col in df.columns:
        if col in MODEL_DEFINITIONS and col != 'EtPMon':
            model_cols.append(col)
        elif col.startswith('ANN_'):
            model_cols.append(col)
            
    for model in model_cols:
        if model in df:
            est = df[model].values
            mean_errors[model] = calculate_errors(obs, est)
            
    errors_df = pd.DataFrame(mean_errors, index=error_names).transpose()
    errors_df = errors_df.reset_index().rename(columns={'index': 'Modelo_Col'})
    errors_df['Modelo'] = errors_df['Modelo_Col'].apply(get_model_name)
    errors_df = errors_df[['Modelo', 'MSE (mm²/día²)', 'RRMSE', 'MAE (mm/día)', 'R²', 'AARE']]
    return errors_df.dropna().sort_values(by='MAE (mm/día)')


# =========================================================================
# 3. LAYOUT
# =========================================================================
app.layout = html.Div([
    html.H1('Análisis Interactivo de ET₀ en Baleares (v2 con ANN)', style={**font_style, 'textAlign': 'center', 'marginBottom': '20px'}),
    html.P('Explora los resultados de ET₀ (empíricos y ANN) para estaciones SIAR. Los datos de ANN se detectan automáticamente.', 
           style={**font_style, 'marginBottom': '20px'}),
    
    dcc.Tabs(id='tabs', value='general', children=[
        dcc.Tab(label='Datos Generales', value='general', style=font_style, selected_style={**font_style, 'backgroundColor': '#e8f0fe'}),
        dcc.Tab(label='Análisis por Estación', value='estaciones', style=font_style, selected_style={**font_style, 'backgroundColor': '#e8f0fe'}),
        dcc.Tab(label='Análisis ANN (Arquitecturas)', value='ann', style=font_style, selected_style={**font_style, 'backgroundColor': '#e8f0fe'}),
    ]),
    
    html.Div(id='tabs-content', style={'padding': '20px'}),
    html.Div(id='error-message', style={'color': 'red', 'fontWeight': 'bold', 'marginTop': '10px'}),
])

# =========================================================================
# 4. CALLBACKS
# =========================================================================

# Callback para actualizar contenido según pestaña
@app.callback(
    [Output('tabs-content', 'children'), Output('error-message', 'children')],
    [Input('tabs', 'value')]
)
def render_tab_content(tab):
    error_msg = ''
    
    if df_all.empty:
        error_msg = '🚨 No hay datos disponibles. Revise que los archivos IBXX_et0_variants.csv existan en la carpeta "datos_siar_baleares" y se hayan cargado correctamente.'
        return html.Div('No datos disponibles.'), error_msg

    if tab == 'general':
        
        # --- Pestaña General ---
        # Calcular errores globales dinámicamente
        general_errors_df = calculate_global_errors(df_all)
        
        # Tabla AHC por estación
        ahc_table = ahc_df[['Estacion', 'AHC_Hargreaves', 'AHC_Valiantzas']].rename(
            columns={'Estacion': 'Estación', 'AHC_Hargreaves': 'AHC Hargreaves', 'AHC_Valiantzas': 'AHC Valiantzas'}
        )
        
        content = [
            html.H2('Datos Generales', style=header_style),
            html.P('Resumen de errores (media de todas las estaciones), coeficientes AHC, y comparación con el TFG (~2020).', 
                   style=font_style),
            
            html.H3('Errores Globales (Python, Datos Cargados)', style=header_style),
            html.P('Calculado dinámicamente de todos los archivos CSV (incluyendo ANNs).', style=font_style),
            dash_table.DataTable(
                data=general_errors_df.to_dict('records'),
                columns=[{'name': col, 'id': col} for col in general_errors_df.columns],
                style_cell=table_style,
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                style_table={'marginBottom': '20px'},
                sort_action="native",
            ),
            
            html.H3('Coeficientes AHC por Estación', style=header_style),
            dash_table.DataTable(
                data=ahc_table.to_dict('records'),
                columns=[{'name': col, 'id': col} for col in ahc_table.columns],
                style_cell=table_style,
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                style_table={'marginBottom': '20px'},
            ),
            
            html.H3('Comparación de Errores: Python vs TFG (Hardcoded)', style=header_style),
            dash_table.DataTable(
                data=tfg_df.to_dict('records'),
                columns=[{'name': col, 'id': col} for col in tfg_df.columns],
                style_cell=table_style,
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                style_table={'marginBottom': '20px'},
            ),
        ]
        return content, error_msg
    
    elif tab == 'estaciones':
        # --- Pestaña Estaciones ---
        if not estaciones_disponibles:
            error_msg = '🚨 No hay datos de estaciones disponibles para seleccionar.'
            return html.Div("No hay estaciones disponibles."), error_msg
            
        content = [
            html.H2('Análisis por Estación', style=header_style),
            html.P('Selecciona una estación para ver metadatos, errores, medias de ET₀, y gráficos interactivos.', style=font_style),
            dcc.Dropdown(
                id='estacion-dropdown',
                options=[{'label': code, 'value': code} for code in estaciones_disponibles], 
                value=estaciones_disponibles[0], # Valor inicial
                style={'marginBottom': '20px', 'width': '50%'}
            ),
            # Contenedor para los resultados de la estación
            dcc.Loading(
                id="loading-estacion",
                type="circle",
                children=html.Div(id='estacion-content')
            )
        ]
        return content, error_msg

    elif tab == 'ann':
        # --- Pestaña Análisis ANN (Nueva) ---
        if not estaciones_disponibles:
            error_msg = '🚨 No hay datos de estaciones disponibles para seleccionar.'
            return html.Div("No hay estaciones disponibles."), error_msg
        
        # Encontrar todas las columnas ANN disponibles en el dataframe global
        ann_cols = sorted([col for col in df_all.columns if col.startswith('ANN_')])
        if not ann_cols:
            return html.Div("No se encontraron columnas de modelos ANN (que empiecen con 'ANN_') en los archivos CSV."), error_msg

        content = [
            html.H2('Análisis de Arquitecturas ANN', style=header_style),
            html.P('Compara el rendimiento de diferentes modelos ANN (distintas arquitecturas, neuronas, etc.) por estación.', style=font_style),
            html.P("Asegúrate de que tus CSVs contengan los resultados con nombres de columna descriptivos (ej. 'ANN_HR_10_neuronas', 'ANN_HR_20_neuronas').", style=font_style),
            
            dbc.Row([
                dbc.Col([
                    html.H5('1. Seleccionar Estación', style=font_style),
                    dcc.Dropdown(
                        id='ann-estacion-dropdown',
                        options=[{'label': code, 'value': code} for code in estaciones_disponibles], 
                        value=estaciones_disponibles[0],
                    ),
                ], width=6),
                dbc.Col([
                    html.H5('2. Seleccionar Modelos ANN a Comparar', style=font_style),
                    dcc.Dropdown(
                        id='ann-model-dropdown',
                        options=[{'label': col, 'value': col} for col in ann_cols],
                        value=[ann_cols[0]] if ann_cols else [],
                        multi=True
                    ),
                ], width=6),
            ]),
            
            dcc.Loading(
                id="loading-ann",
                type="circle",
                children=html.Div(id='ann-analysis-content', style={'marginTop': '20px'})
            )
        ]
        return content, error_msg
    
    return html.Div("Selecciona una pestaña."), error_msg


# Callback para contenido de ESTACIONES individuales
@app.callback(
    [Output('estacion-content', 'children'), Output('error-message', 'children', allow_duplicate=True)],
    [Input('estacion-dropdown', 'value')],
    prevent_initial_call=True
)
def update_estacion_content(code):
    error_msg = ''
    
    if not code:
        return html.Div("Selecciona una estación."), ''
    
    # FILTRAR df_all GLOBALMENTE
    df = df_all[df_all['Estacion'] == code].copy()
    
    if df.empty:
        error_msg = f"No hay datos válidos disponibles para la estación {code} en el DataFrame consolidado."
        return html.Div(error_msg), error_msg
    
    try:
        df = df.dropna(subset=['Fecha'])
        
        if df.empty:
            error_msg = f'No datos válidos en la estación {code} después de la limpieza.'
            return html.Div(error_msg), error_msg
        
        # --- Metadatos ---
        station_info = estaciones_df[estaciones_df['Codigo'] == code]
        station_info = station_info.iloc[0] if not station_info.empty else pd.Series({})
        
        info_data = [
            {'Métrica': 'Nombre', 'Valor': station_info.get('Estacion', 'N/A')},
            {'Métrica': 'Latitud', 'Valor': station_info.get('Latitud', 'N/A')},
            {'Métrica': 'Altitud (m)', 'Valor': station_info.get('Altitud', 'N/A')},
            {'Métrica': 'TempMedia Anual (°C)', 'Valor': round(df['TempMedia'].mean(), 2) if 'TempMedia' in df else 'N/A'},
            {'Métrica': 'Filas Totales', 'Valor': len(df)}
        ]
        info_columns = [{'name': 'Métrica', 'id': 'Métrica'}, {'name': 'Valor', 'id': 'Valor'}]
        
        # --- Tabla medias ET0 (Dinámica) ---
        et0_means = {'Modelo': [], 'Media Anual (mm/día)': []}
        
        # Identificar todos los modelos (empíricos y ANN)
        all_model_cols = []
        for col in df.columns:
            if col in MODEL_DEFINITIONS or col.startswith('ANN_'):
                all_model_cols.append(col)

        for col in all_model_cols:
            if col in df:
                et0_means['Modelo'].append(get_model_name(col))
                et0_means['Media Anual (mm/día)'].append(round(df[col].mean(), 3))
                
        et0_df = pd.DataFrame(et0_means).dropna().sort_values(by='Media Anual (mm/día)')
        et0_columns = [{'name': col, 'id': col} for col in et0_df.columns]
        
        # --- Tabla errores (Dinámica) ---
        obs_col = 'EtPMon'
        error_names = ['MSE (mm²/día²)', 'RRMSE', 'MAE (mm/día)', 'R²', 'AARE']
        mean_errors = {}
        
        if obs_col in df:
            obs = df[obs_col].values
            # Usar la misma lista de modelos de antes (menos la referencia)
            for model in all_model_cols:
                if model != obs_col and model in df:
                    est = df[model].values
                    mean_errors[model] = calculate_errors(obs, est)
        
        errors_df = pd.DataFrame(mean_errors, index=error_names).transpose()
        errors_df = errors_df.reset_index().rename(columns={'index': 'Modelo_Col'})
        errors_df['Modelo'] = errors_df['Modelo_Col'].apply(get_model_name)
        errors_df = errors_df[['Modelo', 'MSE (mm²/día²)', 'RRMSE', 'MAE (mm/día)', 'R²', 'AARE']]
        errors_columns = [{'name': col, 'id': col} for col in errors_df.columns]
        
        # --- Gráficos (Dinámicos) ---
        df_plot = df.copy()
        
        # Columnas de diferencia
        diff_cols_map = {}
        for col in all_model_cols:
            if col != obs_col:
                diff_cols_map[col] = f'{get_model_name(col)} vs SIAR'

        diff_plot_cols = []
        if obs_col in df_plot.columns:
            for est_col, name in diff_cols_map.items():
                if est_col in df_plot.columns:
                    diff_col_name = f'Diff_{est_col}'
                    df_plot[diff_col_name] = df_plot[est_col] - df_plot[obs_col]
                    diff_plot_cols.append((diff_col_name, name)) # Guardar tupla (col, nombre)

        df_plot = df_plot.dropna(subset=['Fecha'])
        
        
        # 1. Serie temporal
        time_cols = [col for col in all_model_cols if col in df_plot.columns]
        df_time_melt = df_plot.melt(id_vars=['Fecha'], value_vars=time_cols,
                                    var_name='Modelo_Col', value_name='ET₀ (mm/día)')
        df_time_melt['Modelo'] = df_time_melt['Modelo_Col'].apply(get_model_name)
        fig_time = px.line(df_time_melt, x='Fecha', y='ET₀ (mm/día)', color='Modelo',
                           title=f'Serie Temporal ET₀ - {code}', render_mode='webgl')
        fig_time.update_traces(opacity=0.7)
        
        # 2. Diferencias vs Temp / 3. Diferencias vs Radiación
        fig_diff_temp = {}
        fig_diff_rs = {}
        
        if diff_plot_cols:
            # Crear un dataframe largo para diferencias
            df_diff_list = []
            for diff_col, nice_name in diff_plot_cols:
                temp_df = df_plot[['Fecha', 'TempMedia', 'Radiacion', diff_col]].copy()
                temp_df['Diferencia'] = nice_name
                temp_df = temp_df.rename(columns={diff_col: 'Valor'})
                df_diff_list.append(temp_df)
            
            df_diff_melt = pd.concat(df_diff_list)

            # vs Temp
            if 'TempMedia' in df_diff_melt.columns:
                df_temp = df_diff_melt.dropna(subset=['TempMedia', 'Valor'])
                fig_diff_temp = px.scatter(df_temp, x='TempMedia', y='Valor', color='Diferencia', trendline='lowess',
                                           title=f'Diferencias vs Temp Media (°C) - {code}', opacity=0.7)
            
            # vs Radiación
            if 'Radiacion' in df_diff_melt.columns:
                df_rs = df_diff_melt.dropna(subset=['Radiacion', 'Valor'])
                fig_diff_rs = px.scatter(df_rs, x='Radiacion', y='Valor', color='Diferencia', trendline='lowess',
                                         title=f'Diferencias vs Radiación (MJ/m²) - {code}', opacity=0.7)
        
        # 4. Diferencias mensuales
        fig_diff_month = {}
        if diff_plot_cols and 'Fecha' in df_plot.columns:
            df_plot['Mes'] = df_plot['Fecha'].dt.month
            
            # Obtener solo las columnas de diff (ej. 'Diff_ET0_calc')
            diff_col_names_only = [col for col, name in diff_plot_cols]
            df_monthly = df_plot.groupby('Mes')[diff_col_names_only].mean().reset_index()
            
            df_monthly = df_monthly.melt(id_vars='Mes', value_vars=diff_col_names_only, 
                                         var_name='Diferencia_Col', value_name='Diff Media Mensual')
            
            # Mapear de 'Diff_ET0_calc' a 'PM Estándar vs SIAR'
            col_to_name_map = {col: name for col, name in diff_plot_cols}
            df_monthly['Diferencia'] = df_monthly['Diferencia_Col'].map(col_to_name_map)
            df_monthly['Mes'] = df_monthly['Mes'].astype(str)
            
            fig_diff_month = px.bar(df_monthly, x='Mes', y='Diff Media Mensual', color='Diferencia', barmode='group',
                                    title=f'Diferencias Mensuales Media - {code}')
        
        # Construcción del contenido
        content = [
            html.H3(f'Información de la Estación {code}', style=header_style),
            dash_table.DataTable(
                data=info_data, columns=info_columns, style_cell=table_style,
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                style_table={'marginBottom': '20px', 'maxWidth': '400px'},
            ),
            
            html.H3('Medias Anuales de ET₀', style=header_style),
            dash_table.DataTable(
                data=et0_df.to_dict('records'), columns=et0_columns, style_cell=table_style,
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                style_table={'marginBottom': '20px'}, sort_action="native"
            ),
            
            html.H3('Errores de Modelos (vs SIAR EtPMon)', style=header_style),
            dash_table.DataTable(
                data=errors_df.to_dict('records'), columns=errors_columns, style_cell=table_style,
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                style_table={'marginBottom': '20px'}, sort_action="native", sort_by=[{'column_id': 'MAE (mm/día)', 'direction': 'asc'}]
            ),
            
            html.H3('Serie Temporal de ET₀', style=header_style),
            dcc.Graph(figure=fig_time),
            
            html.H3('Diferencias vs Temperatura Media', style=header_style),
            dcc.Graph(figure=fig_diff_temp) if fig_diff_temp else html.P('Datos insuficientes.', style=font_style),
            
            html.H3('Diferencias vs Radiación', style=header_style),
            dcc.Graph(figure=fig_diff_rs) if fig_diff_rs else html.P('Datos insuficientes.', style=font_style),
            
            html.H3('Diferencias Mensuales', style=header_style),
            dcc.Graph(figure=fig_diff_month) if fig_diff_month else html.P('Datos insuficientes.', style=font_style),
        ]
        return content, error_msg
    
    except Exception as e:
        error_msg = f"🚨 Error crítico procesando la estación {code}: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc() # Imprimir el stack trace completo en la consola
        return html.Div(error_msg), error_msg

# --- Callback para Pestaña "Análisis ANN" ---
@app.callback(
    Output('ann-analysis-content', 'children'),
    [Input('ann-estacion-dropdown', 'value'),
     Input('ann-model-dropdown', 'value')]
)
def update_ann_analysis(code, selected_models):
    if not code or not selected_models:
        return html.P("Por favor, selecciona una estación y al menos un modelo ANN.")

    # Filtrar el dataframe global por la estación
    df = df_all[df_all['Estacion'] == code].copy()
    
    if df.empty:
        return html.P(f"No hay datos para la estación {code}.")

    # --- Calcular Errores para modelos seleccionados ---
    obs_col = 'EtPMon'
    error_names = ['MSE (mm²/día²)', 'RRMSE', 'MAE (mm/día)', 'R²', 'AARE']
    mean_errors = {}
    
    if obs_col in df:
        obs = df[obs_col].values
        for model in selected_models:
            if model in df:
                est = df[model].values
                mean_errors[model] = calculate_errors(obs, est)
            else:
                print(f"Advertencia: El modelo {model} no se encontró en los datos de la estación {code}")
    else:
        return html.P(f"No se encontró la columna de referencia '{obs_col}' para la estación {code}.")

    errors_df = pd.DataFrame(mean_errors, index=error_names).transpose()
    errors_df = errors_df.reset_index().rename(columns={'index': 'Modelo'})
    errors_columns = [{'name': col, 'id': col} for col in errors_df.columns]
    
    # --- Gráfico de Serie Temporal de Modelos ANN ---
    # Incluir la referencia 'EtPMon' para comparar
    cols_to_plot = selected_models + [obs_col]
    
    # Asegurarse de que todas las columnas existen
    cols_to_plot = [col for col in cols_to_plot if col in df.columns]

    df_plot = df[cols_to_plot + ['Fecha']].copy()
    
    df_melt = df_plot.melt(id_vars=['Fecha'], value_vars=cols_to_plot,
                           var_name='Modelo_Col', value_name='ET₀ (mm/día)')
    df_melt['Modelo'] = df_melt['Modelo_Col'].apply(get_model_name)
    
    fig_time = px.line(df_melt, x='Fecha', y='ET₀ (mm/día)', color='Modelo',
                       title=f'Comparación de Arquitecturas ANN vs SIAR - {code}', render_mode='webgl')
    fig_time.update_traces(opacity=0.8)

    # --- Contenido de la pestaña ---
    content = [
        html.H3(f'Comparativa de Errores ANN para {code}', style=header_style),
        dash_table.DataTable(
            data=errors_df.to_dict('records'),
            columns=errors_columns,
            style_cell=table_style,
            style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
            style_table={'marginBottom': '20px'},
            sort_action="native",
            sort_by=[{'column_id': 'MAE (mm/día)', 'direction': 'asc'}]
        ),
        
        html.H3(f'Serie Temporal de Modelos ANN Seleccionados - {code}', style=header_style),
        dcc.Graph(figure=fig_time)
    ]
    
    return content


if __name__ == '__main__':
    # Usar debug=True para desarrollo local
    # host='0.0.0.0' es necesario para despliegues en servidores
    app.run_server(debug=True, host='0.0.0.0', port=8050)