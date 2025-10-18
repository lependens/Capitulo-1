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
estaciones = ['General'] + [f'IB{str(i).zfill(2)}' for i in range(1, 12)]  # Añadido "General"

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
     Output('time-series', 'figure'), Output('diff-vs-temp', 'figure'), Output('diff-vs-rs', 'figure'), Output('diff-mensual', 'figure')],
    [Input('estacion-dropdown', 'value')]
)
def update_all(code):
    if code == 'General':
        # Cálculo general (agregado de todas estaciones)
        df_all = pd.DataFrame()
        for est in [f'IB{str(i).zfill(2)}' for i in range(1, 12)]:
            file = os.path.join(data_path, f'{est}_et0_variants.csv')
            if os.path.exists(file):
                df = pd.read_csv(file)
                df['Estacion'] = est
                df_all = pd.concat([df_all, df])
        
        if df_all.empty:
            empty_table = [], []
            empty_fig = px.scatter(title='No datos disponibles para General')
            return empty_table, empty_table, empty_table, empty_table, empty_fig, empty_fig, empty_fig, empty_fig
        
        # Tabla info general (medias agregadas)
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
                round(df_all['EtPMon'].mean(), 3),
                round(df_all['ET0_calc'].mean(), 3),
                round(df_all['ET0_sun'].mean(), 3),
                round(df_all['ET0_harg'].mean(), 3),
                round(df_all['ET0_val'].mean(), 3)
            ]
        }
        et0_df = pd.DataFrame(et0_means)
        et0_columns = [{"name": i, "id": i} for i in et0_df.columns]
        et0_data = et0_df.to_dict('records')
        
        # Gráficos general (ej. medias mensuales agregadas)
        df_all['Mes'] = df_all['Fecha'].dt.month
        diff_cols = ['diff', 'diff_sun', 'diff_harg', 'diff_val']
        df_monthly = df_all.groupby('Mes')[diff_cols].mean().reset_index()
        df_monthly = df_monthly.melt(id_vars='Mes', value_vars=diff_cols, var_name='Modelo', value_name='Diff Media Mensual')
        df_monthly['Mes'] = df_monthly['Mes'].astype(str)
        fig_diff_month = px.bar(df_monthly, x='Mes', y='Diff Media Mensual', color='Modelo', barmode='group', title='Diferencias Mensuales Media (Todas Estaciones)')
        
        # Para otros gráficos en 'General', usamos agregados
        fig_time = px.line(df_all.groupby(df_all['Fecha'].dt.year)[['EtPMon', 'ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val']].mean().reset_index(), x='Fecha', y=['EtPMon', 'ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val'], title='Medias Anuales ET₀ (General)')
        
        fig_diff_temp = px.scatter(df_all, x='TempMedia', y='diff', color='Estacion', title='Diferencias vs Temp Media (General)', opacity=0.7)
        
        fig_diff_rs = px.scatter(df_all, x='Radiacion', y='diff', color='Estacion', title='Diferencias vs Radiación (General)', opacity=0.7)
        
        return info_data, info_columns, et0_data, et0_columns, fig_time, fig_diff_temp, fig_diff_rs, fig_diff_month
    
    # Código para estaciones individuales (igual al anterior)
    # ... (pega el código original de update_all para estaciones individuales aquí) ...
    
    # Para "General", usa el código de arriba

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)  # Solo para desarrollo local