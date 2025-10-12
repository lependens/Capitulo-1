import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import os

app = dash.Dash(__name__)

data_path = r'C:/Users/josep/OneDrive/Documentos/GitHub/Capitulo-1/datos_siar_baleares'
estaciones = [f'IB{str(i).zfill(2)}' for i in range(1, 12)]

app.layout = html.Div([
    html.H1('Análisis ET₀ por Estación'),
    dcc.Dropdown(id='estacion-dropdown', options=[{'label': code, 'value': code} for code in estaciones], value='IB05'),
    dcc.Graph(id='time-series'),
    dcc.Graph(id='scatter-pm'),
    dcc.Graph(id='hist-diff')
])

@app.callback(
    [Output('time-series', 'figure'), Output('scatter-pm', 'figure'), Output('hist-diff', 'figure')],
    [Input('estacion-dropdown', 'value')]
)
def update_graphs(code):
    file = os.path.join(data_path, f'{code}_et0_variants.csv')
    if not os.path.exists(file):
        print(f"Archivo no encontrado: {file}")  # Debug en consola
        return px.line(title='Archivo no encontrado'), px.scatter(title='Archivo no encontrado'), px.histogram(title='Archivo no encontrado')
    
    try:
        df = pd.read_csv(file)
        print(f"Cargado {file} con columnas: {df.columns.tolist()}")  # Debug: verifica columnas
        df['Fecha'] = pd.to_datetime(df['Fecha'])  # Asegura Fecha datetime
        
        # Verifica columnas requeridas
        required = ['Fecha', 'EtPMon', 'ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val', 'diff', 'diff_sun', 'diff_harg', 'diff_val']
        missing = [col for col in required if col not in df.columns]  # Corregido: [col for col in required ...
        if missing:
            print(f"Columnas faltantes en {file}: {missing}")
            return px.line(title=f'Faltan columnas: {missing}'), px.scatter(), px.histogram()
        
        df = df.dropna(subset=required[1:])  # Drop NaN en ET0 cols
        
        fig_time = px.line(df, x='Fecha', y=['EtPMon', 'ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val'], title=f'Serie Temporal - {code}')
        fig_scatter = px.scatter(df, x='EtPMon', y='ET0_calc', trendline='lowess', trendline_options=dict(frac=0.1), title=f'PM vs SIAR - {code}')
        df_diff = df[['diff', 'diff_sun', 'diff_harg', 'diff_val']].melt(var_name='Modelo', value_name='Diferencia')
        fig_hist = px.histogram(df_diff, x='Diferencia', color='Modelo', nbins=50, title='Histograma Diferencias')
        
        return fig_time, fig_scatter, fig_hist
    
    except Exception as e:
        print(f"Error procesando {file}: {e}")  # Debug error
        return px.line(title=f'Error: {e}'), px.scatter(), px.histogram()

if __name__ == '__main__':
    app.run(debug=True)  # Ejecuta servidor