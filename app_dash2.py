import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import pandas as pd
import os
import numpy as np

app = dash.Dash(__name__)

data_path = r'C:/Users/josep/OneDrive/Documentos/GitHub/Capitulo-1/datos_siar_baleares'
estaciones = [f'IB{str(i).zfill(2)}' for i in range(1, 12)]
estaciones_df = pd.read_csv(os.path.join(data_path, 'estaciones_baleares.csv'), sep=',', encoding='utf-8-sig', quotechar='"', engine='python')
estaciones_df['Codigo'] = estaciones_df['Codigo'].astype(str).str.strip().str.upper()

# Estilo fuente más visual (Arial, tamaño legible)
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
    html.P('Diferencias (mm/día) por modelo vs TempMedia (°C). Muestra sesgos en condiciones térmicas (ej. sobreestimación en calor). Trendline lowess. Interactivo: hover para valores.', style=font_style),
    dcc.Graph(id='diff-vs-temp'),
    
    html.H2('Diferencias vs Radiación', style=font_style),
    html.P('Diferencias (mm/día) vs Radiación (MJ/m²). Revela dependencia de Rs medida (ej. errores en días nublados). Trendline lowess. Clic leyenda para filtrar.', style=font_style),
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
    file = os.path.join(data_path, f'{code}_et0_variants.csv')
    if not os.path.exists(file):
        empty_table = [], []
        empty_fig = px.scatter(title='Archivo no encontrado')
        return empty_table, empty_table, empty_table, empty_table, empty_fig, empty_fig, empty_fig, empty_fig
    
    try:
        df = pd.read_csv(file)
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        
        # Tabla info estación
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
        
        # Tabla medias ET0
        et0_means = {
            'Modelo': ['SIAR (EtPMon)', 'PM Estándar', 'PM Cielo Claro', 'Hargreaves', 'Valiantzas'],
            'Media Anual (mm/día)': [
                round(df['EtPMon'].mean(), 3) if 'EtPMon' in df.columns else np.nan,
                round(df['ET0_calc'].mean(), 3) if 'ET0_calc' in df.columns else np.nan,
                round(df['ET0_sun'].mean(), 3) if 'ET0_sun' in df.columns else np.nan,
                round(df['ET0_harg'].mean(), 3) if 'ET0_harg' in df.columns else np.nan,
                round(df['ET0_val'].mean(), 3) if 'ET0_val' in df.columns else np.nan
            ]
        }
        et0_df = pd.DataFrame(et0_means)
        et0_columns = [{"name": i, "id": i} for i in et0_df.columns]
        et0_data = et0_df.to_dict('records')
        
        # Limpieza para gráficos
        required = ['Fecha', 'EtPMon', 'ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val', 'diff', 'diff_sun', 'diff_harg', 'diff_val', 'TempMedia', 'Radiacion']
        df_plot = df.dropna(subset=required)
        
        # Serie temporal como puntos
        fig_time = px.scatter(df_plot, x='Fecha', y=['EtPMon', 'ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val'], 
                              title=f'Serie Temporal ET₀ (Puntos) - {code}',
                              opacity=0.6, size_max=3)
        
        # Diferencias vs TempMedia
        df_diff_temp = df_plot.melt(id_vars=['TempMedia'], value_vars=['diff', 'diff_sun', 'diff_harg', 'diff_val'],
                                    var_name='Modelo', value_name='Diferencia')
        fig_diff_temp = px.scatter(df_diff_temp, x='TempMedia', y='Diferencia', color='Modelo', trendline='lowess',
                                   title=f'Diferencias vs Temp Media - {code}', opacity=0.7)
        
        # Diferencias vs Radiacion
        df_diff_rs = df_plot.melt(id_vars=['Radiacion'], value_vars=['diff', 'diff_sun', 'diff_harg', 'diff_val'],
                                  var_name='Modelo', value_name='Diferencia')
        fig_diff_rs = px.scatter(df_diff_rs, x='Radiacion', y='Diferencia', color='Modelo', trendline='lowess',
                                 title=f'Diferencias vs Radiación - {code}', opacity=0.7)
        
        # Diferencias mensuales
        df_diff_month = df_plot.copy()
        df_diff_month['Mes'] = df_diff_month['Fecha'].dt.month
        diff_cols = ['diff', 'diff_sun', 'diff_harg', 'diff_val']
        df_monthly = df_diff_month.groupby('Mes')[diff_cols].mean().reset_index()
        df_monthly = df_monthly.melt(id_vars='Mes', value_vars=diff_cols, var_name='Modelo', value_name='Diff Media Mensual')
        df_monthly['Mes'] = df_monthly['Mes'].astype(str)
        fig_diff_month = px.bar(df_monthly, x='Mes', y='Diff Media Mensual', color='Modelo', barmode='group', title='Diferencias Mensuales Media (Todos Años)')
        
        return info_data, info_columns, et0_data, et0_columns, fig_time, fig_diff_temp, fig_diff_rs, fig_diff_month
    
    except Exception as e:
        print(f"Error: {e}")
        empty_table = [], []
        empty_fig = px.scatter(title=f'Error: {e}')
        return empty_table, empty_table, empty_table, empty_table, empty_fig, empty_fig, empty_fig, empty_fig

if __name__ == '__main__':
    app.run(debug=True)