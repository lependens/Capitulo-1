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

app.layout = html.Div([
    html.H1('Análisis ET₀ por Estación'),
    dcc.Dropdown(id='estacion-dropdown', options=[{'label': code, 'value': code} for code in estaciones], value='IB05'),
    
    html.H2('Información General de la Estación'),
    dash_table.DataTable(id='tabla-info-estacion', columns=[], data=[]),
    
    html.H2('Medias Anuales de ET₀ por Modelo'),
    dash_table.DataTable(id='tabla-medias-et0', columns=[], data=[]),
    
    html.H2('Serie Temporal de ET₀'),
    dcc.Graph(id='time-series'),
    
    html.H2('Scatter PM vs SIAR'),
    dcc.Graph(id='scatter-pm'),
    
    html.H2('Diferencias Mensuales Media (Todos Años)'),
    dcc.Graph(id='diff-mensual')
])

@app.callback(
    [Output('tabla-info-estacion', 'data'), Output('tabla-info-estacion', 'columns'),
     Output('tabla-medias-et0', 'data'), Output('tabla-medias-et0', 'columns'),
     Output('time-series', 'figure'), Output('scatter-pm', 'figure'), Output('diff-mensual', 'figure')],
    [Input('estacion-dropdown', 'value')]
)
def update_all(code):
    file = os.path.join(data_path, f'{code}_et0_variants.csv')
    if not os.path.exists(file):
        empty_table = [], []
        empty_fig = px.line(title='Archivo no encontrado')
        return empty_table, empty_table, empty_fig, empty_fig, empty_fig
    
    try:
        df = pd.read_csv(file)
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        
        # Tabla info estación (de catálogo + stats datos)
        station_info = estaciones_df[estaciones_df['Codigo'] == code].iloc[0]
        info_data = [
            {'Métrica': 'Nombre', 'Valor': station_info['Estacion']},
            {'Métrica': 'Latitud', 'Valor': station_info['Latitud']},
            {'Métrica': 'Altitud (m)', 'Valor': station_info['Altitud']},
            {'Métrica': 'TempMedia Anual (°C)', 'Valor': round(df['TempMedia'].mean(), 2)},
            {'Métrica': 'VelViento Media (m/s)', 'Valor': round(df['VelViento'].mean(), 2)},
            {'Métrica': 'HumedadMedia Anual (%)', 'Valor': round(df['HumedadMedia'].mean(), 2)},
            {'Métrica': 'Radiacion Media (MJ/m²)', 'Valor': round(df['Radiacion'].mean(), 2)},
            {'Métrica': 'Filas Totales', 'Valor': len(df)}
        ]
        info_columns = [{"name": 'Métrica', "id": 'Métrica'}, {"name": 'Valor', "id": 'Valor'}]
        
        # Tabla medias ET0
        et0_means = {
            'Modelo': ['SIAR (EtPMon)', 'PM Estándar', 'PM Cielo Claro', 'Hargreaves', 'Valiantzas'],
            'Media Anual (mm/día)': [
                round(df['EtPMon'].mean(), 3),
                round(df['ET0_calc'].mean(), 3),
                round(df['ET0_sun'].mean(), 3),
                round(df['ET0_harg'].mean(), 3),
                round(df['ET0_val'].mean(), 3)
            ]
        }
        et0_df = pd.DataFrame(et0_means)
        et0_columns = [{"name": i, "id": i} for i in et0_df.columns]
        et0_data = et0_df.to_dict('records')
        
        # Gráficos existentes
        required = ['Fecha', 'EtPMon', 'ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val']
        df_plot = df.dropna(subset=required)
        
        fig_time = px.line(df_plot, x='Fecha', y=['EtPMon', 'ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val'], title=f'Serie Temporal - {code}')
        fig_scatter = px.scatter(df_plot, x='EtPMon', y='ET0_calc', trendline='lowess', trendline_options=dict(frac=0.1), title=f'PM vs SIAR - {code}')
        
        # Nuevo: Diferencias mensuales media (todos años)
        df_diff_month = df_plot.copy()
        df_diff_month['Mes'] = df_diff_month['Fecha'].dt.month
        diff_cols = ['diff', 'diff_sun', 'diff_harg', 'diff_val']
        df_monthly = df_diff_month.groupby('Mes')[diff_cols].mean().reset_index()
        df_monthly = df_monthly.melt(id_vars='Mes', value_vars=diff_cols, var_name='Modelo', value_name='Diff Media Mensual')
        df_monthly['Mes'] = df_monthly['Mes'].astype(str)  # Para eje categorizado
        fig_diff_month = px.bar(df_monthly, x='Mes', y='Diff Media Mensual', color='Modelo', barmode='group', title='Diferencias Mensuales Media (Todos Años)')
        
        return info_data, info_columns, et0_data, et0_columns, fig_time, fig_scatter, fig_diff_month
    
    except Exception as e:
        print(f"Error: {e}")
        empty_table = [], []
        empty_fig = px.line(title=f'Error: {e}')
        return empty_table, empty_table, empty_table, empty_table, empty_fig, empty_fig, empty_fig

if __name__ == '__main__':
    app.run(debug=True)