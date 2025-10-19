import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore') # Opcional: para limpiar logs de warnings de pandas/numpy

# Inicializar la app con tema Bootstrap para diseño moderno
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Esencial para Render/Gunicorn

# =========================================================================
# 1. CARGA GLOBAL DE DATOS (La Corrección Principal para Render)
# =========================================================================
data_path = 'datos_siar_baleares'
estaciones = ['IB01', 'IB02', 'IB03', 'IB04', 'IB05'] # Extensible

font_style = {'family': 'Arial, sans-serif', 'size': 14, 'color': '#333333'}
header_style = {'fontWeight': 'bold', 'color': '#2c3e50', 'marginBottom': '10px'}
table_style = {'textAlign': 'left', 'fontFamily': 'Arial', 'fontSize': '14px'}

def load_data_globally():
    """Carga y concatena todos los archivos de datos de ET0."""
    df_all = pd.DataFrame()
    found_estaciones = []
    
    print("Iniciando carga global de archivos CSV...")
    
    for est in estaciones:
        file = os.path.join(data_path, f'{est}_et0_variants.csv')
        if os.path.exists(file):
            try:
                # Usar 'utf-8-sig' para manejar el Byte Order Mark (BOM) que a veces tienen los CSV de Excel
                df = pd.read_csv(file, encoding='utf-8-sig', encoding_errors='replace', dtype={'Estacion': str})
                # Intentar inferir el formato de fecha, útil si hay diferentes formatos
                df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce', dayfirst=True) 
                df['Estacion'] = est
                df_all = pd.concat([df_all, df], ignore_index=True)
                found_estaciones.append(est)
                print(f"Cargada estación {est}: {len(df)} filas.")
            except Exception as e:
                print(f"Error al cargar {file}: {e}")
        else:
            print(f"Archivo no encontrado: {file}")
                
    if df_all.empty:
        print("ADVERTENCIA: df_all está vacío. Los gráficos no funcionarán.")
    
    df_all = df_all.dropna(subset=['Fecha']).sort_values(by='Fecha')
    return df_all, found_estaciones

# Carga de datos de ET0
try:
    df_all, estaciones_disponibles = load_data_globally()
except FileNotFoundError as e:
    print(f"Error fatal de carga: {e}")
    df_all = pd.DataFrame()
    estaciones_disponibles = []


# Cargar metadatos de estaciones
estaciones_csv_path = os.path.join(data_path, 'estaciones_baleares.csv')
estaciones_df = pd.DataFrame()
if os.path.exists(estaciones_csv_path):
    try:
        estaciones_df = pd.read_csv(estaciones_csv_path, sep=',', encoding='utf-8-sig', quotechar='"', engine='python', encoding_errors='replace')
        # Limpieza de código para asegurar coincidencia con los datos
        estaciones_df['Codigo'] = estaciones_df['Codigo'].astype(str).str.strip().str.upper() 
        print(f"Cargados metadatos de {len(estaciones_df)} estaciones.")
    except Exception as e:
        print(f"Error al cargar metadatos de estaciones: {e}")
else:
    print(f"Advertencia: Archivo {estaciones_csv_path} no encontrado.")

# Cargar AHC (asumiendo ahc_por_estacion.csv)
ahc_csv_path = os.path.join(data_path, 'ahc_por_estacion.csv')
ahc_df = pd.DataFrame(columns=['Estacion', 'AHC_Hargreaves', 'AHC_Valiantzas'])
if os.path.exists(ahc_csv_path):
    try:
        ahc_df = pd.read_csv(ahc_csv_path, sep=',', encoding='utf-8-sig')
        # Limpieza de código para asegurar coincidencia
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
    # Manejar NaN. Solo considerar pares no-NaN
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
        pass # r2 se mantiene como nan si falla el calculo
            
    aare = np.nan
    # Para AARE, se necesita obs != 0
    valid_aare = valid & (obs != 0)
    obs_aare, est_aare = obs[valid_aare], est[valid_aare]
    if len(obs_aare) > 0:
        aare = np.mean(np.abs((obs_aare - est_aare) / obs_aare))
        
    # RRMSE = RMSE / mean(obs)
    rrmse_val = rmse / np.mean(obs) if np.mean(obs) != 0 else np.nan
    
    return round(mse, 3), round(rrmse_val, 3), round(mae, 3), round(r2, 3), round(aare, 3)


# =========================================================================
# 3. LAYOUT
# =========================================================================
app.layout = html.Div([
    html.H1('Análisis Interactivo de ET₀ en Baleares', style={**font_style, 'textAlign': 'center', 'marginBottom': '20px'}),
    html.P('Explora los resultados de evapotranspiración (ET₀) para estaciones SIAR en Baleares. Selecciona entre datos generales o por estación.', 
           style={**font_style, 'marginBottom': '20px'}),
    
    dcc.Tabs(id='tabs', value='general', children=[
        dcc.Tab(label='Datos Generales', value='general', style=font_style, selected_style={**font_style, 'backgroundColor': '#e8f0fe'}),
        dcc.Tab(label='Estaciones', value='estaciones', style=font_style, selected_style={**font_style, 'backgroundColor': '#e8f0fe'}),
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
    
    if tab == 'general':
        # USAR df_all CARGADO GLOBALMENTE
        if df_all.empty:
            error_msg = '🚨 No hay datos disponibles. Revise que los archivos IBXX_et0_variants.csv existan en la carpeta "datos_siar_baleares" y se hayan cargado correctamente.'
            return html.Div('No datos disponibles.'), error_msg
        
        # Tabla errores generales (datos hardcoded basados en docs_1.3)
        general_errors = {
            'Modelo': ['PM Estándar', 'PM Cielo Claro', 'Hargreaves', 'Valiantzas', 'Hargreaves Ajustado', 'Valiantzas Ajustado'],
            'MSE (mm²/día²)': [0.00, 0.48, 0.56, 0.24, 0.31, 0.21],
            'RRMSE': [0.02, 0.22, 0.24, 0.16, 0.18, 0.15],
            'MAE (mm/día)': [0.02, 0.46, 0.61, 0.37, 0.43, 0.35],
            'R²': [1.00, 0.94, 0.91, 0.94, 0.91, 0.94],
            'AARE': [0.02, 0.16, 0.29, 0.16, 0.18, 0.14]
        }
        general_errors_df = pd.DataFrame(general_errors)
        
        # Tabla AHC por estación
        ahc_table = ahc_df[['Estacion', 'AHC_Hargreaves', 'AHC_Valiantzas']].rename(
            columns={'Estacion': 'Estación', 'AHC_Hargreaves': 'AHC Hargreaves', 'AHC_Valiantzas': 'AHC Valiantzas'}
        )
        
        # Contenido pestaña General
        content = [
            html.H2('Datos Generales', style=header_style),
            html.P('Resumen de errores generales (media de IB01-IB05), coeficientes AHC por estación, y comparación con el TFG (11 estaciones, datos hasta ~2020).', 
                   style=font_style),
            
            html.H3('Errores Generales (Python, IB01-IB05)', style=header_style),
            dash_table.DataTable(
                data=general_errors_df.to_dict('records'),
                columns=[{'name': col, 'id': col} for col in general_errors_df.columns],
                style_cell=table_style,
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                style_table={'marginBottom': '20px'},
            ),
            
            html.H3('Coeficientes AHC por Estación', style=header_style),
            html.P('Valores de AHC calculados para Hargreaves y Valiantzas por estación (usados en los modelos ajustados).', style=font_style),
            dash_table.DataTable(
                data=ahc_table.to_dict('records'),
                columns=[{'name': col, 'id': col} for col in ahc_table.columns],
                style_cell=table_style,
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                style_table={'marginBottom': '20px'},
            ),
            
            html.H3('Comparación de Errores: Python (2024) vs TFG (~2020)', style=header_style),
            dash_table.DataTable(
                data=tfg_df.to_dict('records'),
                columns=[{'name': col, 'id': col} for col in tfg_df.columns],
                style_cell=table_style,
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                style_table={'marginBottom': '20px'},
            ),
        ]
        return content, error_msg
    
    else:
        # Pestaña Estaciones
        
        # Manejo de error si no hay estaciones disponibles
        if not estaciones_disponibles:
            error_msg = '🚨 No hay datos de estaciones disponibles para seleccionar.'
            return html.Div("No hay estaciones disponibles."), error_msg
            
        content = [
            html.H2('Análisis por Estación', style=header_style),
            html.P('Selecciona una estación para ver metadatos, errores, medias de ET₀, y gráficos interactivos.', style=font_style),
            dcc.Dropdown(
                id='estacion-dropdown',
                # USAR estaciones_disponibles
                options=[{'label': code, 'value': code} for code in estaciones_disponibles], 
                value=estaciones_disponibles[0], # Valor inicial
                style={'marginBottom': '20px', 'width': '50%'}
            ),
            html.Div(id='estacion-content'),
        ]
        return content, error_msg

# Callback para contenido de estaciones individuales
@app.callback(
    [Output('estacion-content', 'children'), Output('error-message', 'children')],
    [Input('estacion-dropdown', 'value')]
)
def update_estacion_content(code):
    error_msg = ''
    
    if not code:
        return html.Div("Selecciona una estación."), ''
    
    # 🌟 FILTRAR df_all GLOBALMENTE: Evita errores de acceso a disco en Render.
    df = df_all[df_all['Estacion'] == code].copy()
    
    if df.empty:
        error_msg = f"No hay datos válidos disponibles para la estación {code} en el DataFrame consolidado."
        return html.Div(error_msg), error_msg
    
    try:
        df = df.dropna(subset=['Fecha'])
        
        if df.empty:
            error_msg = f'No datos válidos en la estación {code} después de la limpieza.'
            return html.Div(error_msg), error_msg
        
        # Metadatos
        station_info = estaciones_df[estaciones_df['Codigo'] == code]
        station_info = station_info.iloc[0] if not station_info.empty else pd.Series({})
        
        info_data = [
            {'Métrica': 'Nombre', 'Valor': station_info.get('Estacion', 'N/A')},
            {'Métrica': 'Latitud', 'Valor': station_info.get('Latitud', 'N/A')},
            {'Métrica': 'Altitud (m)', 'Valor': station_info.get('Altitud', 'N/A')},
            {'Métrica': 'TempMedia Anual (°C)', 'Valor': round(df['TempMedia'].mean(), 2) if 'TempMedia' in df else 'N/A'},
            {'Métrica': 'VelViento Media (m/s)', 'Valor': round(df['VelViento'].mean(), 2) if 'VelViento' in df else 'N/A'},
            {'Métrica': 'HumedadMedia Anual (%)', 'Valor': round(df['HumedadMedia'].mean(), 2) if 'HumedadMedia' in df else 'N/A'},
            {'Métrica': 'Radiacion Media (MJ/m²)', 'Valor': round(df['Radiacion'].mean(), 2) if 'Radiacion' in df else 'N/A'},
            {'Métrica': 'Filas Totales', 'Valor': len(df)}
        ]
        info_columns = [{'name': 'Métrica', 'id': 'Métrica'}, {'name': 'Valor', 'id': 'Valor'}]
        
        # Tabla medias ET0
        et0_means = {
            'Modelo': ['SIAR (EtPMon)', 'PM Estándar', 'PM Cielo Claro', 'Hargreaves', 'Valiantzas', 'Hargreaves Ajustado', 'Valiantzas Ajustado'],
            'Media Anual (mm/día)': [
                round(df['EtPMon'].mean(), 3) if 'EtPMon' in df else np.nan,
                round(df['ET0_calc'].mean(), 3) if 'ET0_calc' in df else np.nan,
                round(df['ET0_sun'].mean(), 3) if 'ET0_sun' in df else np.nan,
                round(df['ET0_harg'].mean(), 3) if 'ET0_harg' in df else np.nan,
                round(df['ET0_val'].mean(), 3) if 'ET0_val' in df else np.nan,
                round(df['ET0_harg_aj'].mean(), 3) if 'ET0_harg_aj' in df else np.nan,
                round(df['ET0_val_aj'].mean(), 3) if 'ET0_val_aj' in df else np.nan
            ]
        }
        et0_df = pd.DataFrame(et0_means).dropna(subset=['Media Anual (mm/día)'])
        et0_columns = [{'name': col, 'id': col} for col in et0_df.columns]
        
        # Tabla errores
        obs_col = 'EtPMon'
        models = ['ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val', 'ET0_harg_aj', 'ET0_val_aj']
        error_names = ['MSE (mm²/día²)', 'RRMSE', 'MAE (mm/día)', 'R²', 'AARE']
        mean_errors = {}
        
        if obs_col in df:
            obs = df[obs_col].values
            for model in models:
                if model in df:
                    est = df[model].values
                    # Usar la función calculate_errors definida globalmente
                    mean_errors[model] = calculate_errors(obs, est)
        
        errors_df = pd.DataFrame(mean_errors, index=error_names).transpose()
        errors_df = errors_df.reset_index().rename(columns={'index': 'Modelo'})
        errors_columns = [{'name': 'Modelo', 'id': 'Modelo'}] + [{'name': col, 'id': col} for col in error_names]
        
        # Gráficos: Calcular las diferencias (asumiendo EtPMon es la referencia)
        required_plot_cols = ['Fecha', 'EtPMon', 'ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val', 'ET0_harg_aj', 'ET0_val_aj', 'TempMedia', 'Radiacion']
        df_plot = df[[col for col in required_plot_cols if col in df.columns]].copy()
        
        # Calcular columnas de diferencia (solo si EtPMon existe)
        diff_cols_map = {
            'ET0_calc': 'PM Estándar vs SIAR',
            'ET0_sun': 'PM Cielo Claro vs SIAR',
            'ET0_harg': 'Hargreaves vs SIAR',
            'ET0_val': 'Valiantzas vs SIAR',
            'ET0_harg_aj': 'Hargreaves Aj. vs SIAR',
            'ET0_val_aj': 'Valiantzas Aj. vs SIAR',
        }
        
        diff_plot_cols = []
        if 'EtPMon' in df_plot.columns:
            for est_col, name in diff_cols_map.items():
                if est_col in df_plot.columns:
                    diff_col_name = f'Diff_{name.replace(" ", "_")}'
                    df_plot[diff_col_name] = df_plot[est_col] - df_plot['EtPMon']
                    diff_plot_cols.append(diff_col_name)

        df_plot = df_plot.dropna(subset=['Fecha'])
        
        
        # --- 1. Serie temporal ---
        time_cols = [col for col in ['EtPMon', 'ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val', 'ET0_harg_aj', 'ET0_val_aj'] if col in df_plot.columns]
        df_time_melt = df_plot.melt(id_vars=['Fecha'], value_vars=time_cols,
                                    var_name='Modelo', value_name='ET₀ (mm/día)')
        fig_time = px.line(df_time_melt, x='Fecha', y='ET₀ (mm/día)', color='Modelo',
                              title=f'Serie Temporal ET₀ - {code}', render_mode='webgl') # webgl para rendimiento
        fig_time.update_traces(opacity=0.6)
        
        # --- 2. Diferencias vs Temp / 3. Diferencias vs Radiación ---
        fig_diff_temp = {}
        fig_diff_rs = {}
        
        if diff_plot_cols:
            df_diff_melt = df_plot.melt(value_vars=diff_plot_cols, var_name='Diferencia', value_name='Valor')
            df_diff_melt['Diferencia'] = df_diff_melt['Diferencia'].str.replace('Diff_', '').str.replace('_vs_SIAR', '').str.replace('_', ' ')
            
            # Diferencias vs Temp
            if 'TempMedia' in df_plot.columns:
                df_temp = pd.concat([df_plot['TempMedia'], df_diff_melt['Valor'], df_diff_melt['Diferencia']], axis=1).dropna()
                fig_diff_temp = px.scatter(df_temp, x='TempMedia', y='Valor', color='Diferencia', trendline='lowess',
                                        title=f'Diferencias vs Temp Media (°C) - {code}', opacity=0.7)
            
            # Diferencias vs Radiación
            if 'Radiacion' in df_plot.columns:
                df_rs = pd.concat([df_plot['Radiacion'], df_diff_melt['Valor'], df_diff_melt['Diferencia']], axis=1).dropna()
                fig_diff_rs = px.scatter(df_rs, x='Radiacion', y='Valor', color='Diferencia', trendline='lowess',
                                        title=f'Diferencias vs Radiación (MJ/m²) - {code}', opacity=0.7)
        
        
        # --- 4. Diferencias mensuales ---
        fig_diff_month = {}
        if diff_plot_cols and 'Fecha' in df_plot.columns:
            df_plot['Mes'] = df_plot['Fecha'].dt.month
            df_monthly = df_plot.groupby('Mes')[diff_plot_cols].mean().reset_index()
            df_monthly = df_monthly.melt(id_vars='Mes', value_vars=diff_plot_cols, var_name='Diferencia', value_name='Diff Media Mensual')
            df_monthly['Mes'] = df_monthly['Mes'].astype(str)
            df_monthly['Diferencia'] = df_monthly['Diferencia'].str.replace('Diff_', '').str.replace('_vs_SIAR', '').str.replace('_', ' ')
            
            fig_diff_month = px.bar(df_monthly, x='Mes', y='Diff Media Mensual', color='Diferencia', barmode='group',
                                    title=f'Diferencias Mensuales Media - {code}')
        
        # Construcción del contenido
        content = [
            html.H3(f'Información de la Estación {code}', style=header_style),
            dash_table.DataTable(
                data=info_data,
                columns=info_columns,
                style_cell=table_style,
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                style_table={'marginBottom': '20px', 'maxWidth': '400px'},
            ),
            
            html.H3('Medias Anuales de ET₀', style=header_style),
            dash_table.DataTable(
                data=et0_df.to_dict('records'),
                columns=et0_columns,
                style_cell=table_style,
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                style_table={'marginBottom': '20px'},
            ),
            
            html.H3('Errores de Modelos', style=header_style),
            dash_table.DataTable(
                data=errors_df.to_dict('records'),
                columns=errors_columns,
                style_cell=table_style,
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                style_table={'marginBottom': '20px'},
            ),
            
            html.H3('Serie Temporal de ET₀', style=header_style),
            dcc.Graph(figure=fig_time),
            
            html.H3('Diferencias vs Temperatura Media', style=header_style),
            dcc.Graph(figure=fig_diff_temp) if fig_diff_temp else html.P('Datos insuficientes para el gráfico.', style=font_style),
            
            html.H3('Diferencias vs Radiación', style=header_style),
            dcc.Graph(figure=fig_diff_rs) if fig_diff_rs else html.P('Datos insuficientes para el gráfico.', style=font_style),
            
            html.H3('Diferencias Mensuales', style=header_style),
            dcc.Graph(figure=fig_diff_month) if fig_diff_month else html.P('Datos insuficientes para el gráfico.', style=font_style),
        ]
        return content, error_msg
    
    except Exception as e:
        error_msg = f"🚨 Error crítico procesando la estación {code}: {str(e)}"
        print(error_msg)
        return html.Div(error_msg), error_msg

if __name__ == '__main__':
    # Usar el argumento host='0.0.0.0' para despliegues en servidores como Render
    # En producción, gunicorn manejará esta parte.
    app.run_server(debug=True, host='0.0.0.0', port=8050)