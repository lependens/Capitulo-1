import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import pandas as pd
import os
import numpy as np

app = dash.Dash(__name__)

# Exponer el objeto WSGI para Gunicorn
server = app.server  # Esto es lo que Gunicorn usará

# Ruta relativa para Render
data_path = 'datos_siar_baleares'
estaciones = ['General'] + [f'IB{str(i).zfill(2)}' for i in range(1, 12)]

# Verificar si el archivo estaciones_baleares.csv existe
estaciones_csv_path = os.path.join(data_path, 'estaciones_baleares.csv')
if not os.path.exists(estaciones_csv_path):
    raise FileNotFoundError(f"No se encontró el archivo: {estaciones_csv_path}")

estaciones_df = pd.read_csv(estaciones_csv_path, sep=',', encoding='utf-8-sig', quotechar='"', engine='python')
estaciones_df['Codigo'] = estaciones_df['Codigo'].astype(str).str.strip().str.upper()

font_style = dict(family="Arial, sans-serif", size=14, color="#333333")

app.layout = html.Div([
    html.H1('Análisis ET₀ por Estación', style=font_style),
    dcc.Dropdown(id='estacion-dropdown', options=[{'label': code, 'value': code} for code in estaciones], value='IB05'),
    
    html.H2('Información General de la Estación', style=font_style),
    html.P('Resumen de metadatos de la estación (de catálogo SIAR) y estadísticas anuales medias de variables clave en los datos procesados. Útil para contexto climático.', style=font_style),
    dash_table.DataTable(id='tabla-info-estacion', columns=[], data=[], style_cell={'textAlign': 'left', 'fontFamily': 'Arial'}),
    
    html.H2('Medias Anuales de ET₀ por Modelo', style=font_style),
    html.P('Media anual de ET₀ (mm/día) para cada modelo vs SIAR. Compara precisión global. Clic en leyenda para activar/desactivar modelos.', style=font_style),
    dash_table.DataTable(id='tabla-medias-et0', columns=[], data=[], style_cell={'textAlign': 'left', 'fontFamily': 'Arial'}),
    
    html.H2('Tabla de Errores de Modelos (Agregados para General)', style=font_style),
    html.P('Errores (MSE, RMSE, MAE, R², AARE) para cada modelo vs SIAR. Solo para "General" (media de todas estaciones). MSE/RMSE/MAE/AARE en mm/día, R² sin unidades.', style=font_style),
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
     Output('time-series', 'figure'), Output('diff-vs-temp', 'figure'), Output('diff-vs-rs', 'figure'), Output('diff-mensual', 'figure')],
    [Input('estacion-dropdown', 'value')]
)
def update_all(code):
    if code == 'General':
        # Carga agregada
        df_all = pd.DataFrame()
        for est in [f'IB{str(i).zfill(2)}' for i in range(1, 12)]:
            file = os.path.join(data_path, f'{est}_et0_variants.csv')
            if os.path.exists(file):
                df = pd.read_csv(file)
                df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')  # Asegura datetime por estación
                df['Estacion'] = est
                df_all = pd.concat([df_all, df], ignore_index=True)
        
        if df_all.empty:
            empty_table = [], []
            empty_fig = px.scatter(title='No datos disponibles para General')
            return empty_table, empty_table, empty_table, empty_table, empty_table, empty_table, empty_fig, empty_fig, empty_fig, empty_fig
        
        # Re-aplicar datetime después de concat (fix error .dt)
        df_all['Fecha'] = pd.to_datetime(df_all['Fecha'], errors='coerce')
        df_all = df_all.dropna(subset=['Fecha'])  # Drop filas con Fecha NaT
        
        # Tabla info general (agregados)
        info_data = [
            {'Métrica': 'Número de Estaciones', 'Valor': len(df_all['Estacion'].unique())},
            {'Métrica': 'TempMedia Anual (°C)', 'Valor': round(df_all['TempMedia'].mean(), 2)},
            {'Métrica': 'VelViento Media (m/s)', 'Valor': round(df_all['VelViento'].mean(), 2)},
            {'Métrica': 'HumedadMedia Anual (%)', 'Valor': round(df_all['HumedadMedia'].mean(), 2)},
            {'Métrica': 'Radiacion Media (MJ/m²)', 'Valor': round(df_all['Radiacion'].mean(), 2)},
            {'Métrica': 'Filas Totales', 'Valor': len(df_all)}
        ]
        info_columns = [{"name": 'Métrica', "id": 'Métrica'}, {"name": 'Valor', "id": 'Valor'}]
        
        # Tabla medias ET0 general
        et0_means = {
            'Modelo': ['SIAR (EtPMon)', 'PM Estándar', 'PM Cielo Claro', 'Hargreaves', 'Valiantzas'],
            'Media Anual (mm/día)': [
                round(df_all['EtPMon'].mean(), 2),  # Reducido a 2 decimales
                round(df_all['ET0_calc'].mean(), 2),
                round(df_all['ET0_sun'].mean(), 2),
                round(df_all['ET0_harg'].mean(), 2),
                round(df_all['ET0_val'].mean(), 2)
            ]
        }
        et0_df = pd.DataFrame(et0_means)
        et0_columns = [{"name": i, "id": i} for i in et0_df.columns]
        et0_data = et0_df.to_dict('records')
        
        # Tabla errores general
        def calculate_errors(obs, est):
            if len(obs) != len(est) or len(obs) == 0:
                return np.nan, np.nan, np.nan, np.nan, np.nan
            
            mse = np.mean((obs - est)**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(obs - est))
            if len(obs) > 1 and np.std(obs) > 0 and np.std(est) > 0:
                r2 = np.corrcoef(obs, est)[0,1]**2
            else:
                r2 = np.nan
            aare = np.mean(np.abs((obs - est) / obs)) if np.all(obs != 0) else np.nan
            return round(mse, 2), round(rmse, 2), round(mae, 2), round(r2, 2), round(aare, 2)  # Reducido a 2 decimales
        
        obs = df_all['EtPMon'].values
        models = ['ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val']
        error_names = ['MSE', 'RMSE', 'MAE', 'R2', 'AARE']
        mean_errors = {}
        for model in models:
            est = df_all[model].values
            mean_errors[model] = calculate_errors(obs, est)
        
        errors_df = pd.DataFrame(mean_errors, index=error_names).transpose()
        errors_columns = [{"name": i, "id": i} for i in errors_df.columns]
        errors_data = errors_df.to_dict('records')
        
        # Gráficos general (ej. medias mensuales agregadas)
        df_all['Mes'] = df_all['Fecha'].dt.month
        diff_cols = ['diff', 'diff_sun', 'diff_harg', 'diff_val']
        df_monthly = df_all.groupby('Mes')[diff_cols].mean().reset_index()
        df_monthly = df_monthly.melt(id_vars='Mes', value_vars=diff_cols, var_name='Modelo', value_name='Diff Media Mensual')
        df_monthly['Mes'] = df_monthly['Mes'].astype(str)
        fig_diff_month = px.bar(df_monthly, x='Mes', y='Diff Media Mensual', color='Modelo', barmode='group', title='Diferencias Mensuales Media (Todas Estaciones)')
        
        fig_time = px.line(df_all.groupby(df_all['Fecha'].dt.year)[['EtPMon', 'ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val']].mean().reset_index(), x='Fecha', y=['EtPMon', 'ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val'], title='Medias Anuales ET₀ (General)')
        
        fig_diff_temp = px.scatter(df_all, x='TempMedia', y='diff', color='Estacion', title='Diferencias vs Temp Media (General)', opacity=0.7)
        
        fig_diff_rs = px.scatter(df_all, x='Radiacion', y='diff', color='Estacion', title='Diferencias vs Radiación (General)', opacity=0.7)
        
        return info_data, info_columns, et0_data, et0_columns, errors_data, errors_columns, fig_time, fig_diff_temp, fig_diff_rs, fig_diff_month
    
    # Código para estaciones individuales (igual al anterior)
    file = os.path.join(data_path, f'{code}_et0_variants.csv')
    if not os.path.exists(file):
        print(f"Archivo no encontrado: {file}")
        empty_table = [], []
        empty_fig = px.scatter(title=f'Archivo no encontrado: {file}')
        return empty_table, empty_table, empty_table, empty_table, empty_table, empty_table, empty_fig, empty_fig, empty_fig, empty_fig
    
    try:
        df = pd.read_csv(file)
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df = df.dropna(subset=['Fecha'])  # Drop filas con Fecha NaT
        
        station_info = estaciones_df[estaciones_df['Codigo'] == code].iloc[0]
        info_data = [
            {'Métrica': 'Nombre', 'Valor': station_info['Estacion']},
            {'Métrica': 'Latitud', 'Valor': station_info['Latitud']},
            {'Métrica': 'Altitud (m)', 'Valor': station_info['Altitud']},
            {'Métrica': 'TempMedia Anual (°C)', 'Valor': round(df['TempMedia'].mean(), 2) if 'TempMedia' in df.columns else 'N/A'},
            {'Métrica': 'VelViento Media (m/s)', 'Valor': round(df['VelViento'].mean(), 2) if 'VelViento' in df.columns else 'N/A'},
            {'Métrica': 'HumedadMedia Anual (%)', 'Valor': round(df['HumedadMedia'].mean(), 2) if 'HumedadMedia' in df.columns else 'N/A'},
            {'Métrica': 'Radiacion Media (MJ/m²)', 'Valor': round(df['Radiacion'].mean(), 2) if 'Radiacion' in df.columns else 'N/A'},
            {'Métrica': 'Filas Totales', 'Valor': len(df)}
        ]
        info_columns = [{"name": 'Métrica', "id": 'Métrica'}, {"name": 'Valor', "id": 'Valor'}]
        
        et0_means = {
            'Modelo': ['SIAR (EtPMon)', 'PM Estándar', 'PM Cielo Claro', 'Hargreaves', 'Valiantzas'],
            'Media Anual (mm/día)': [
                round(df['EtPMon'].mean(), 2),  # Reducido a 2 decimales
                round(df['ET0_calc'].mean(), 2),
                round(df['ET0_sun'].mean(), 2),
                round(df['ET0_harg'].mean(), 2),
                round(df['ET0_val'].mean(), 2)
            ]
        }
        et0_df = pd.DataFrame(et0_means)
        et0_columns = [{"name": i, "id": i} for i in et0_df.columns]
        et0_data = et0_df.to_dict('records')
        
        # Tabla errores vacía para individuales
        errors_data = []
        errors_columns = []
        
        required = ['Fecha', 'EtPMon', 'ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val', 'diff', 'diff_sun', 'diff_harg', 'diff_val', 'TempMedia', 'Radiacion']
        df_plot = df.dropna(subset=required)
        print(f"Filas para gráficos después de drop: {len(df_plot)}")
        
        fig_time = px.scatter(df_plot, x='Fecha', y=['EtPMon', 'ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val'], 
                              title=f'Serie Temporal ET₀ (Puntos) - {code}',
                              opacity=0.6, size_max=3)
        
        df_diff_temp = df_plot.melt(id_vars=['TempMedia'], value_vars=['diff', 'diff_sun', 'diff_harg', 'diff_val'],
                                    var_name='Modelo', value_name='Diferencia')
        fig_diff_temp = px.scatter(df_diff_temp, x='TempMedia', y='Diferencia', color='Modelo', trendline='lowess',
                                   title=f'Diferencias vs Temp Media - {code}', opacity=0.7)
        
        df_diff_rs = df_plot.melt(id_vars=['Radiacion'], value_vars=['diff', 'diff_sun', 'diff_harg', 'diff_val'],
                                  var_name='Modelo', value_name='Diferencia')
        fig_diff_rs = px.scatter(df_diff_rs, x='Radiacion', y='Diferencia', color='Modelo', trendline='lowess',
                                 title=f'Diferencias vs Radiación - {code}', opacity=0.7)
        
        df_diff_month = df_plot.copy()
        df_diff_month['Mes'] = df_diff_month['Fecha'].dt.month
        diff_cols = ['diff', 'diff_sun', 'diff_harg', 'diff_val']
        df_monthly = df_diff_month.groupby('Mes')[diff_cols].mean().reset_index()
        df_monthly = df_monthly.melt(id_vars='Mes', value_vars=diff_cols, var_name='Modelo', value_name='Diff Media Mensual')
        df_monthly['Mes'] = df_monthly['Mes'].astype(str)
        fig_diff_month = px.bar(df_monthly, x='Mes', y='Diff Media Mensual', color='Modelo', barmode='group', title='Diferencias Mensuales Media (Todos Años)')
        
        return info_data, info_columns, et0_data, et0_columns, errors_data, errors_columns, fig_time, fig_diff_temp, fig_diff_rs, fig_diff_month
    
    except Exception as e:
        print(f"Error: {e}")
        empty_table = [], []
        empty_fig = px.scatter(title=f'Error: {e}')
        return empty_table, empty_table, empty_table, empty_table, empty_table, empty_table, empty_fig, empty_fig, empty_fig, empty_fig

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)  # Solo para desarrollo local