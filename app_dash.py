import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import os

# Inicializar la app con tema Bootstrap para diseño moderno
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Para Render/Gunicorn

# Ruta de datos
data_path = 'datos_siar_baleares'
estaciones = ['IB01', 'IB02', 'IB03', 'IB04', 'IB05']  # Extensible a IB06-IB11
font_style = {'family': 'Arial, sans-serif', 'size': 14, 'color': '#333333'}
header_style = {'fontWeight': 'bold', 'color': '#2c3e50', 'marginBottom': '10px'}
table_style = {'textAlign': 'left', 'fontFamily': 'Arial', 'fontSize': '14px'}

# Cargar metadatos de estaciones
estaciones_csv_path = os.path.join(data_path, 'estaciones_baleares.csv')
if not os.path.exists(estaciones_csv_path):
    raise FileNotFoundError(f"No se encontró: {estaciones_csv_path}")
estaciones_df = pd.read_csv(estaciones_csv_path, sep=',', encoding='utf-8-sig', quotechar='"', engine='python', encoding_errors='replace')
estaciones_df['Codigo'] = estaciones_df['Codigo'].astype(str).str.strip().str.upper()

# Cargar AHC (asumiendo ahc_por_estacion.csv generado por calculate_ahc.py)
ahc_csv_path = os.path.join(data_path, 'ahc_por_estacion.csv')
if os.path.exists(ahc_csv_path):
    ahc_df = pd.read_csv(ahc_csv_path, sep=',', encoding='utf-8-sig')
    ahc_df['Estacion'] = ahc_df['Estacion'].astype(str).str.strip().str.upper()
else:
    ahc_df = pd.DataFrame(columns=['Estacion', 'AHC_Hargreaves', 'AHC_Valiantzas'])

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

# Layout con pestañas
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

# Callback para actualizar contenido según pestaña
@app.callback(
    [Output('tabs-content', 'children'), Output('error-message', 'children')],
    [Input('tabs', 'value')]
)
def render_tab_content(tab):
    error_msg = ''
    
    if tab == 'general':
        # Cargar datos generales
        df_all = pd.DataFrame()
        for est in estaciones:
            file = os.path.join(data_path, f'{est}_et0_variants.csv')
            if os.path.exists(file):
                df = pd.read_csv(file, encoding='utf-8-sig', encoding_errors='replace')
                df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
                df['Estacion'] = est
                df_all = pd.concat([df_all, df], ignore_index=True)
        
        if df_all.empty:
            error_msg = 'No hay datos disponibles para General.'
            return html.Div('No datos disponibles.'), error_msg
        
        df_all = df_all.dropna(subset=['Fecha'])
        
        # Tabla errores generales (de docs_1.3_Análisis errores.md)
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
            
            html.H3('Errores Generales', style=header_style),
            dash_table.DataTable(
                data=general_errors_df.to_dict('records'),
                columns=[{'name': col, 'id': col} for col in general_errors_df.columns],
                style_cell=table_style,
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                style_table={'marginBottom': '20px'},
            ),
            
            html.H3('Coeficientes AHC por Estación', style=header_style),
            html.P('Valores de AHC calculados para Hargreaves y Valiantzas por estación.', style=font_style),
            dash_table.DataTable(
                data=ahc_table.to_dict('records'),
                columns=[{'name': col, 'id': col} for col in ahc_table.columns],
                style_cell=table_style,
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                style_table={'marginBottom': '20px'},
            ),
            
            html.H3('Comparación con TFG', style=header_style),
            html.P('Comparación de errores generales (IB01-IB05) entre Python (2024) y TFG (~2020). HGRs no incluido en Python.', style=font_style),
            dash_table.DataTable(
                data=tfg_df.to_dict('records'),
                columns=[{'name': col, 'id': col} for col in tfg_df.columns],
                style_cell=table_style,
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                style_table={'marginBottom': '20px'},
            ),
            
            html.H3('Análisis de la Comparación', style=header_style),
            html.Ul([
                html.Li('Hargreaves (Sin Ajustar): Python reduce MAE ~9% (0.61 vs. 0.672), MSE ~19%.', style=font_style),
                html.Li('Valiantzas (Sin Ajustar): Python mejora MAE ~20% (0.37 vs. 0.461), MSE ~37%.', style=font_style),
                html.Li('Hargreaves Ajustado: Python reduce MAE ~8% (0.43 vs. 0.468), AARE ~19%.', style=font_style),
                html.Li('Valiantzas Ajustado: Python mejora MAE ~6% (0.35 vs. 0.373), MSE ~16%.', style=font_style),
                html.Li('Causas: Datos hasta 2024 (Python) vs. ~2020 (TFG), filtrado ±3σ, y solo 5 estaciones vs. 11.', style=font_style),
                html.Li('Recomendación: Procesar IB06-IB11 en Python y comparar subperíodo hasta 2020.', style=font_style),
            ], style={'marginBottom': '20px'}),
        ]
        return content, error_msg
    
    else:
        # Pestaña Estaciones
        content = [
            html.H2('Análisis por Estación', style=header_style),
            html.P('Selecciona una estación para ver metadatos, errores, medias de ET₀, y gráficos interactivos.', style=font_style),
            dcc.Dropdown(
                id='estacion-dropdown',
                options=[{'label': code, 'value': code} for code in estaciones],
                value='IB01',
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
    file = os.path.join(data_path, f'{code}_et0_variants.csv')
    
    if not os.path.exists(file):
        error_msg = f"Archivo no encontrado: {file}"
        return html.Div(error_msg), error_msg
    
    try:
        df = pd.read_csv(file, encoding='utf-8-sig', encoding_errors='replace')
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df = df.dropna(subset=['Fecha'])
        
        if df.empty:
            error_msg = f"No datos válidos en {file}"
            return html.Div(error_msg), error_msg
        
        # Metadatos
        station_info = estaciones_df[estaciones_df['Codigo'] == code].iloc[0] if not estaciones_df[estaciones_df['Codigo'] == code].empty else pd.Series({'Estacion': 'N/A', 'Latitud': 'N/A', 'Altitud': 'N/A'})
        info_data = [
            {'Métrica': 'Nombre', 'Valor': station_info['Estacion']},
            {'Métrica': 'Latitud', 'Valor': station_info['Latitud']},
            {'Métrica': 'Altitud (m)', 'Valor': station_info['Altitud']},
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
                round(df['EtPMon'].mean(), 2) if 'EtPMon' in df else np.nan,
                round(df['ET0_calc'].mean(), 2) if 'ET0_calc' in df else np.nan,
                round(df['ET0_sun'].mean(), 2) if 'ET0_sun' in df else np.nan,
                round(df['ET0_harg'].mean(), 2) if 'ET0_harg' in df else np.nan,
                round(df['ET0_val'].mean(), 2) if 'ET0_val' in df else np.nan,
                round(df['ET0_harg_aj'].mean(), 2) if 'ET0_harg_aj' in df else np.nan,
                round(df['ET0_val_aj'].mean(), 2) if 'ET0_val_aj' in df else np.nan
            ]
        }
        et0_df = pd.DataFrame(et0_means)
        et0_columns = [{'name': col, 'id': col} for col in et0_df.columns]
        
        # Tabla errores
        def calculate_errors(obs, est):
            valid = ~np.isnan(obs) & ~np.isnan(est) & (obs != 0)
            obs, est = obs[valid], est[valid]
            if len(obs) < 2:
                return np.nan, np.nan, np.nan, np.nan, np.nan
            mse = np.mean((obs - est)**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(obs - est))
            r2 = np.corrcoef(obs, est)[0,1]**2 if np.std(obs) > 0 and np.std(est) > 0 else np.nan
            aare = np.mean(np.abs((obs - est) / obs)) if np.all(obs != 0) else np.nan
            return round(mse, 2), round(rmse, 2), round(mae, 2), round(r2, 2), round(aare, 2)
        
        obs = df['EtPMon'].values if 'EtPMon' in df else np.array([])
        models = ['ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val', 'ET0_harg_aj', 'ET0_val_aj']
        error_names = ['MSE (mm²/día²)', 'RRMSE', 'MAE (mm/día)', 'R²', 'AARE']
        mean_errors = {}
        for model in models:
            if model in df:
                est = df[model].values
                mean_errors[model] = calculate_errors(obs, est)
        
        errors_df = pd.DataFrame(mean_errors, index=error_names).transpose()
        errors_df = errors_df.reset_index().rename(columns={'index': 'Modelo'})
        errors_columns = [{'name': 'Modelo', 'id': 'Modelo'}] + [{'name': col, 'id': col} for col in error_names]
        
        # Gráficos
        required = ['Fecha', 'EtPMon', 'ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val', 'ET0_harg_aj', 'ET0_val_aj', 'diff', 'diff_sun', 'diff_harg', 'diff_val', 'diff_harg_aj', 'diff_val_aj', 'TempMedia', 'Radiacion']
        existing_req = [col for col in required if col in df.columns]
        df_plot = df.dropna(thresh=len(existing_req)*0.5)
        
        if df_plot.empty:
            error_msg = f'Datos insuficientes para gráficos en {code}. Revisa NaNs.'
            return html.Div(error_msg), error_msg
        
        # Serie temporal
        df_time_melt = df_plot.melt(id_vars=['Fecha'], value_vars=['EtPMon', 'ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val', 'ET0_harg_aj', 'ET0_val_aj'],
                                    var_name='Modelo', value_name='ET₀ (mm/día)')
        fig_time = px.scatter(df_time_melt, x='Fecha', y='ET₀ (mm/día)', color='Modelo',
                              title=f'Serie Temporal ET₀ - {code}', opacity=0.6, hover_data=['Fecha'])
        
        # Diferencias vs Temp
        df_diff_temp = df_plot.melt(id_vars=['TempMedia'], value_vars=['diff', 'diff_sun', 'diff_harg', 'diff_val', 'diff_harg_aj', 'diff_val_aj'],
                                    var_name='Modelo', value_name='Diferencia')
        fig_diff_temp = px.scatter(df_diff_temp, x='TempMedia', y='Diferencia', color='Modelo', trendline='lowess',
                                   title=f'Diferencias vs Temp Media - {code}', opacity=0.7, hover_data=['TempMedia'])
        
        # Diferencias vs Radiación
        df_diff_rs = df_plot.melt(id_vars=['Radiacion'], value_vars=['diff', 'diff_sun', 'diff_harg', 'diff_val', 'diff_harg_aj', 'diff_val_aj'],
                                  var_name='Modelo', value_name='Diferencia')
        fig_diff_rs = px.scatter(df_diff_rs, x='Radiacion', y='Diferencia', color='Modelo', trendline='lowess',
                                 title=f'Diferencias vs Radiación - {code}', opacity=0.7, hover_data=['Radiacion'])
        
        # Diferencias mensuales
        df_plot['Mes'] = df_plot['Fecha'].dt.month
        diff_cols = [col for col in ['diff', 'diff_sun', 'diff_harg', 'diff_val', 'diff_harg_aj', 'diff_val_aj'] if col in df_plot]
        df_monthly = df_plot.groupby('Mes')[diff_cols].mean().reset_index()
        df_monthly = df_monthly.melt(id_vars='Mes', value_vars=diff_cols, var_name='Modelo', value_name='Diff Media Mensual')
        df_monthly['Mes'] = df_monthly['Mes'].astype(str)
        fig_diff_month = px.bar(df_monthly, x='Mes', y='Diff Media Mensual', color='Modelo', barmode='group',
                                title=f'Diferencias Mensuales Media - {code}')
        
        content = [
            html.H3(f'Información de la Estación {code}', style=header_style),
            dash_table.DataTable(
                data=info_data,
                columns=info_columns,
                style_cell=table_style,
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                style_table={'marginBottom': '20px'},
            ),
            
            html.H3('Medias Anuales de ET₀', style=header_style),
            html.P('Media anual de ET₀ (mm/día) por modelo vs SIAR.', style=font_style),
            dash_table.DataTable(
                data=et0_df.to_dict('records'),
                columns=et0_columns,
                style_cell=table_style,
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                style_table={'marginBottom': '20px'},
            ),
            
            html.H3('Errores de Modelos', style=header_style),
            html.P('Errores calculados vs SIAR para cada modelo.', style=font_style),
            dash_table.DataTable(
                data=errors_df.to_dict('records'),
                columns=errors_columns,
                style_cell=table_style,
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                style_table={'marginBottom': '20px'},
            ),
            
            html.H3('Serie Temporal de ET₀', style=header_style),
            html.P('Evolución diaria de ET₀ (mm/día). Interactivo: clic en leyenda, zoom con mouse.', style=font_style),
            dcc.Graph(figure=fig_time),
            
            html.H3('Diferencias vs Temperatura Media', style=header_style),
            html.P('Diferencias (mm/día) vs TempMedia (°C).', style=font_style),
            dcc.Graph(figure=fig_diff_temp),
            
            html.H3('Diferencias vs Radiación', style=header_style),
            html.P('Diferencias (mm/día) vs Radiación (MJ/m²).', style=font_style),
            dcc.Graph(figure=fig_diff_rs),
            
            html.H3('Diferencias Mensuales', style=header_style),
            html.P('Media mensual de diferencias (mm/día) por modelo.', style=font_style),
            dcc.Graph(figure=fig_diff_month),
        ]
        return content, error_msg
    
    except Exception as e:
        error_msg = f"Error procesando {code}: {str(e)}"
        return html.Div(error_msg), error_msg

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)