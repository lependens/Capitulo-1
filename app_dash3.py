import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import pandas as pd
import os
import numpy as np

app = dash.Dash(__name__)

# Exponer el objeto WSGI para Gunicorn/Render
server = app.server

# Ruta relativa para Render
data_path = 'datos_siar_baleares'
estaciones = ['General'] + [f'IB{str(i).zfill(2)}' for i in range(1, 12)]

# Verificar y cargar metadatos de estaciones
estaciones_csv_path = os.path.join(data_path, 'estaciones_baleares.csv')
if not os.path.exists(estaciones_csv_path):
    raise FileNotFoundError(f"No se encontró el archivo: {estaciones_csv_path}")

estaciones_df = pd.read_csv(estaciones_csv_path, sep=',', encoding='utf-8-sig', quotechar='"', engine='python', encoding_errors='replace')
estaciones_df['Codigo'] = estaciones_df['Codigo'].astype(str).str.strip().str.upper()

font_style = dict(family="Arial, sans-serif", size=14, color="#333333")

app.layout = html.Div([
    html.H1('Análisis ET₀ por Estación', style=font_style),
    dcc.Dropdown(id='estacion-dropdown', options=[{'label': code, 'value': code} for code in estaciones], value='IB05'),
    
    html.Div(id='error-message', style={'color': 'red', 'fontWeight': 'bold'}),
    
    html.H2('Información General de la Estación', style=font_style),
    html.P('Resumen de metadatos de la estación (de catálogo SIAR) y estadísticas anuales medias de variables clave en los datos procesados. Útil para contexto climático.', style=font_style),
    dash_table.DataTable(id='tabla-info-estacion', columns=[], data=[], style_cell={'textAlign': 'left', 'fontFamily': 'Arial'}),
    
    html.H2('Medias Anuales de ET₀ por Modelo', style=font_style),
    html.P('Media anual de ET₀ (mm/día) para cada modelo vs SIAR. Compara precisión global. Clic en leyenda para activar/desactivar modelos.', style=font_style),
    dash_table.DataTable(id='tabla-medias-et0', columns=[], data=[], style_cell={'textAlign': 'left', 'fontFamily': 'Arial'}),
    
    html.H2('Tabla de Errores de Modelos', style=font_style),
    html.P('Errores (MSE, RMSE, MAE, R², AARE) para cada modelo vs SIAR. MSE/RMSE/MAE/AARE en mm/día, R² sin unidades.', style=font_style),
    dash_table.DataTable(id='tabla-errores', columns=[], data=[], style_cell={'textAlign': 'left', 'fontFamily': 'Arial'}),
    
    html.H2('Serie Temporal de ET₀ (Puntos)', style=font_style),
    html.P('Evolución diaria de ET₀ (mm/día) como puntos (mejor para datos densos). Leyenda interactiva: clic para ocultar/mostrar modelos, zoom con mouse.', style=font_style),
    dcc.Graph(id='time-series'),
    
    html.H2('Diferencias vs Temperatura Media', style=font_style),
    html.P('Diferencias (mm/día) por modelo vs TempMedia (°C). Muestra sesgos en condiciones térmicas (ej. sobreestimación en calor). Interactivo: hover para valores.', style=font_style),
    dcc.Graph(id='diff-vs-temp'),
    
    html.H2('Diferencias vs Radiación', style=font_style),
    html.P('Diferencias (mm/día) vs Radiación (MJ/m²). Revela dependencia de Rs medida (ej. errores en días nublados). Interactivo: hover para valores.', style=font_style),
    dcc.Graph(id='diff-vs-rs'),
    
    html.H2('Diferencias Mensuales Media (Todos Años)', style=font_style),
    html.P('Media mensual de diferencias (mm/día) agrupadas por mes (1-12, todos años). Barra agrupada: compara estacionalidad de errores. Hover para valores exactos.', style=font_style),
    dcc.Graph(id='diff-mensual')
])

@app.callback(
    [Output('tabla-info-estacion', 'data'), Output('tabla-info-estacion', 'columns'),
     Output('tabla-medias-et0', 'data'), Output('tabla-medias-et0', 'columns'),
     Output('tabla-errores', 'data'), Output('tabla-errores', 'columns'),
     Output('time-series', 'figure'), Output('diff-vs-temp', 'figure'), 
     Output('diff-vs-rs', 'figure'), Output('diff-mensual', 'figure'),
     Output('error-message', 'children')],
    [Input('estacion-dropdown', 'value')]
)
def update_all(code):
    error_msg = ''
    if code == 'General':
        df_all = pd.DataFrame()
        for est in [f'IB{str(i).zfill(2)}' for i in range(1, 12)]:
            file = os.path.join(data_path, f'{est}_et0_variants.csv')
            if os.path.exists(file):
                df = pd.read_csv(file, encoding='utf-8-sig', encoding_errors='replace')
                df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
                df['Estacion'] = est
                df_all = pd.concat([df_all, df], ignore_index=True)
        
        if df_all.empty:
            error_msg = 'No datos disponibles para General'
            empty_table = [], []
            empty_fig = px.scatter(title='No datos disponibles')
            return empty_table, empty_table, empty_table, empty_table, empty_table, empty_table, empty_fig, empty_fig, empty_fig, empty_fig, error_msg
        
        df_all = df_all.dropna(subset=['Fecha'])
        print(f"Filas totales para General: {len(df_all)}")  # Log para debug
        
        # Optimización: Sample si muy grande
        if len(df_all) > 50000:
            df_all = df_all.sample(frac=0.5, random_state=42)
        
        # Tabla info general
        info_data = [
            {'Métrica': 'Número de Estaciones', 'Valor': len(df_all['Estacion'].unique())},
            {'Métrica': 'TempMedia Anual (°C)', 'Valor': round(df_all['TempMedia'].mean(), 2)},
            {'Métrica': 'VelViento Media (m/s)', 'Valor': round(df_all['VelViento'].mean(), 2)},
            {'Métrica': 'HumedadMedia Anual (%)', 'Valor': round(df_all['HumedadMedia'].mean(), 2)},
            {'Métrica': 'Radiacion Media (MJ/m²)', 'Valor': round(df_all['Radiacion'].mean(), 2)},
            {'Métrica': 'Filas Totales', 'Valor': len(df_all)}
        ]
        info_columns = [{"name": 'Métrica', "id": 'Métrica'}, {"name": 'Valor', "id": 'Valor'}]
        
        # Tabla medias ET0
        et0_means = {
            'Modelo': ['SIAR (EtPMon)', 'PM Estándar', 'PM Cielo Claro', 'Hargreaves', 'Valiantzas'],
            'Media Anual (mm/día)': [
                round(df_all['EtPMon'].mean(), 2),
                round(df_all['ET0_calc'].mean(), 2),
                round(df_all['ET0_sun'].mean(), 2),
                round(df_all['ET0_harg'].mean(), 2),
                round(df_all['ET0_val'].mean(), 2)
            ]
        }
        et0_df = pd.DataFrame(et0_means)
        et0_columns = [{"name": i, "id": i} for i in et0_df.columns]
        et0_data = et0_df.to_dict('records')
        
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
        
        obs = df_all['EtPMon'].values
        models = ['ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val']
        error_names = ['MSE', 'RMSE', 'MAE', 'R2', 'AARE']
        mean_errors = {}
        for model in models:
            est = df_all[model].values
            mean_errors[model] = calculate_errors(obs, est)
        
        errors_df = pd.DataFrame(mean_errors, index=error_names).transpose()
        errors_columns = [{"name": 'Modelo', "id": 'index'}] + [{"name": i, "id": i} for i in errors_df.columns]
        errors_df = errors_df.reset_index().rename(columns={'index': 'Modelo'})
        errors_data = errors_df.to_dict('records')
        
        # Gráficos para General
        required = ['Fecha', 'EtPMon', 'ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val', 'diff', 'diff_sun', 'diff_harg', 'diff_val', 'TempMedia', 'Radiacion']
        df_plot = df_all.dropna(thresh=len(required)*0.5)  # Drop si >50% NaNs en fila
        print(f"Filas para gráficos General después de drop: {len(df_plot)}")
        if df_plot.empty:
            error_msg += ' Datos insuficientes para gráficos en General.'
        
        # Downsample para performance: cada 30 días
        df_sample = df_plot.iloc[::30, :]
        
        df_time_melt = df_sample.melt(id_vars=['Fecha'], value_vars=['EtPMon', 'ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val'],
                                       var_name='Modelo', value_name='ET0')
        fig_time = px.scatter(df_time_melt, x='Fecha', y='ET0', color='Modelo', title='Serie Temporal ET₀ (Puntos, Downsampled) - General',
                              opacity=0.6, hover_data=['Estacion'])
        
        df_diff_temp = df_plot.melt(id_vars=['TempMedia', 'Estacion'], value_vars=['diff', 'diff_sun', 'diff_harg', 'diff_val'],
                                    var_name='Modelo', value_name='Diferencia')
        fig_diff_temp = px.scatter(df_diff_temp, x='TempMedia', y='Diferencia', color='Modelo', trendline='lowess',
                                   title='Diferencias vs Temp Media (General)', opacity=0.7, hover_data=['Estacion'])
        
        df_diff_rs = df_plot.melt(id_vars=['Radiacion', 'Estacion'], value_vars=['diff', 'diff_sun', 'diff_harg', 'diff_val'],
                                  var_name='Modelo', value_name='Diferencia')
        fig_diff_rs = px.scatter(df_diff_rs, x='Radiacion', y='Diferencia', color='Modelo', trendline='lowess',
                                 title='Diferencias vs Radiación (General)', opacity=0.7, hover_data=['Estacion'])
        
        df_all['Mes'] = df_all['Fecha'].dt.month
        diff_cols = ['diff', 'diff_sun', 'diff_harg', 'diff_val']
        df_monthly = df_all.groupby('Mes')[diff_cols].mean().reset_index()
        df_monthly = df_monthly.melt(id_vars='Mes', value_vars=diff_cols, var_name='Modelo', value_name='Diff Media Mensual')
        df_monthly['Mes'] = df_monthly['Mes'].astype(str)
        fig_diff_month = px.bar(df_monthly, x='Mes', y='Diff Media Mensual', color='Modelo', barmode='group', title='Diferencias Mensuales Media (Todas Estaciones)')
        
        return info_data, info_columns, et0_data, et0_columns, errors_data, errors_columns, fig_time, fig_diff_temp, fig_diff_rs, fig_diff_month, error_msg
    
    # Para estaciones individuales
    file = os.path.join(data_path, f'{code}_et0_variants.csv')
    if not os.path.exists(file):
        error_msg = f"Archivo no encontrado: {file}"
        empty_table = [], []
        empty_fig = px.scatter(title=error_msg)
        return empty_table, empty_table, empty_table, empty_table, empty_table, empty_table, empty_fig, empty_fig, empty_fig, empty_fig, error_msg
    
    try:
        df = pd.read_csv(file, encoding='utf-8-sig', encoding_errors='replace')
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df = df.dropna(subset=['Fecha'])
        print(f"Filas totales para {code}: {len(df)}")  # Log
        
        if df.empty:
            error_msg = f"No datos válidos en {file}"
            empty_table = [], []
            empty_fig = px.scatter(title=error_msg)
            return empty_table, empty_table, empty_table, empty_table, empty_table, empty_table, empty_fig, empty_fig, empty_fig, empty_fig, error_msg
        
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
        info_columns = [{"name": 'Métrica', "id": 'Métrica'}, {"name": 'Valor', "id": 'Valor'}]
        
        et0_means = {
            'Modelo': ['SIAR (EtPMon)', 'PM Estándar', 'PM Cielo Claro', 'Hargreaves', 'Valiantzas'],
            'Media Anual (mm/día)': [
                round(df['EtPMon'].mean(), 2) if 'EtPMon' in df else np.nan,
                round(df['ET0_calc'].mean(), 2) if 'ET0_calc' in df else np.nan,
                round(df['ET0_sun'].mean(), 2) if 'ET0_sun' in df else np.nan,
                round(df['ET0_harg'].mean(), 2) if 'ET0_harg' in df else np.nan,
                round(df['ET0_val'].mean(), 2) if 'ET0_val' in df else np.nan
            ]
        }
        et0_df = pd.DataFrame(et0_means)
        et0_columns = [{"name": i, "id": i} for i in et0_df.columns]
        et0_data = et0_df.to_dict('records')
        
        # Tabla errores para individual (nueva)
        obs = df['EtPMon'].values if 'EtPMon' in df else np.array([])
        models = ['ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val']
        mean_errors = {}
        for model in models:
            if model in df:
                est = df[model].values
                mean_errors[model] = calculate_errors(obs, est)
        
        errors_df = pd.DataFrame(mean_errors, index=error_names).transpose()
        errors_columns = [{"name": 'Modelo', "id": 'index'}] + [{"name": i, "id": i} for i in errors_df.columns]
        errors_df = errors_df.reset_index().rename(columns={'index': 'Modelo'})
        errors_data = errors_df.to_dict('records')
        
        # Gráficos
        required = ['Fecha', 'EtPMon', 'ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val', 'diff', 'diff_sun', 'diff_harg', 'diff_val', 'TempMedia', 'Radiacion']
        existing_req = [col for col in required if col in df.columns]
        df_plot = df.dropna(thresh=len(existing_req)*0.5)  # Más flexible
        print(f"Filas para gráficos {code} después de drop: {len(df_plot)}")
        if df_plot.empty:
            error_msg += f' Datos insuficientes para gráficos en {code}. Revisa NaNs en columnas clave.'
        
        df_time_melt = df_plot.melt(id_vars=['Fecha'], value_vars=['EtPMon', 'ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val'],
                                       var_name='Modelo', value_name='ET0')
        fig_time = px.scatter(df_time_melt, x='Fecha', y='ET0', color='Modelo', 
                              title=f'Serie Temporal ET₀ (Puntos) - {code}',
                              opacity=0.6, size_max=3, hover_data=['Fecha'])
        
        df_diff_temp = df_plot.melt(id_vars=['TempMedia'], value_vars=['diff', 'diff_sun', 'diff_harg', 'diff_val'],
                                    var_name='Modelo', value_name='Diferencia')
        fig_diff_temp = px.scatter(df_diff_temp, x='TempMedia', y='Diferencia', color='Modelo', trendline='lowess',
                                   title=f'Diferencias vs Temp Media - {code}', opacity=0.7, hover_data=['TempMedia'])
        
        df_diff_rs = df_plot.melt(id_vars=['Radiacion'], value_vars=['diff', 'diff_sun', 'diff_harg', 'diff_val'],
                                  var_name='Modelo', value_name='Diferencia')
        fig_diff_rs = px.scatter(df_diff_rs, x='Radiacion', y='Diferencia', color='Modelo', trendline='lowess',
                                 title=f'Diferencias vs Radiación - {code}', opacity=0.7, hover_data=['Radiacion'])
        
        df_plot['Mes'] = df_plot['Fecha'].dt.month
        diff_cols = [col for col in ['diff', 'diff_sun', 'diff_harg', 'diff_val'] if col in df_plot]
        df_monthly = df_plot.groupby('Mes')[diff_cols].mean().reset_index()
        df_monthly = df_monthly.melt(id_vars='Mes', value_vars=diff_cols, var_name='Modelo', value_name='Diff Media Mensual')
        df_monthly['Mes'] = df_monthly['Mes'].astype(str)
        fig_diff_month = px.bar(df_monthly, x='Mes', y='Diff Media Mensual', color='Modelo', barmode='group', title='Diferencias Mensuales Media (Todos Años)')
        
        return info_data, info_columns, et0_data, et0_columns, errors_data, errors_columns, fig_time, fig_diff_temp, fig_diff_rs, fig_diff_month, error_msg
    
    except Exception as e:
        error_msg = f"Error procesando {code}: {str(e)}"
        empty_table = [], []
        empty_fig = px.scatter(title=error_msg)
        return empty_table, empty_table, empty_table, empty_table, empty_table, empty_table, empty_fig, empty_fig, empty_fig, empty_fig, error_msg

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)  # Para local