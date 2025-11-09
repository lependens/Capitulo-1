import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore') # Opcional: para limpiar logs de warnings de pandas/numpy

# Inicializar la app con tema Bootstrap para diseño moderno
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Esencial para Render/Gunicorn

# =========================================================================
# 1. CARGA GLOBAL DE DATOS (CON FALLBACK Y RUTA ABSOLUTA CORREGIDA)
# =========================================================================
# Definir la ruta base como el directorio donde reside este script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas prioritarias, relativas al directorio del script
data_path_priority = os.path.join(SCRIPT_DIR, 'datos_siar_baleares')
# Rutas secundarias (la raíz del proyecto, en caso de que los CSV estén fuera de la subcarpeta)
data_path_fallback = SCRIPT_DIR

estaciones = ['IB01', 'IB02', 'IB03', 'IB04', 'IB05'] # Extensible

font_style = {'family': 'Arial, sans-serif', 'size': 14, 'color': '#333333'}
header_style = {'fontWeight': 'bold', 'color': '#2c3e50', 'marginBottom': '10px'}
table_style = {'textAlign': 'left', 'fontFamily': 'Arial', 'fontSize': '14px'}

def load_data_globally():
    """Carga y concatena todos los archivos de datos de ET0, intentando varias rutas."""
    df_all = pd.DataFrame()
    found_estaciones = []
    
    print("Iniciando carga global de archivos CSV...")
    print(f"Directorio de script base: {SCRIPT_DIR}")

    # Iterar sobre la ruta prioritaria y luego la de fallback (si es diferente)
    paths_to_check = [data_path_priority]
    if data_path_fallback != data_path_priority:
         paths_to_check.append(data_path_fallback)

    for path in paths_to_check:
        print(f"Intentando cargar desde la ruta: {path}")
        
        # Copiamos la lista para saber qué buscar en esta iteración
        estaciones_a_buscar = [code for code in estaciones if code not in found_estaciones]

        for code in estaciones_a_buscar:
            filename = f'{code}_et0_variants_ajustado.csv'
            filepath = os.path.join(path, filename)
            
            try:
                # Comprobación de existencia para un mensaje de error más limpio
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"El archivo no existe en la ruta.")

                df = pd.read_csv(filepath)
                df['Estacion'] = code
                df['Fecha'] = pd.to_datetime(df['Fecha'])
                
                # Convertir columnas clave a numérico, manejando errores
                for col in ['ET0_calc', 'ET0_harg_ajustado', 'ET0_val_ajustado', 'TempMedia', 'Radiacion']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df_all = pd.concat([df_all, df], ignore_index=True)
                found_estaciones.append(code)
                print(f"✅ Archivo {filename} cargado exitosamente desde {path}.")

            except FileNotFoundError:
                print(f"Advertencia: Archivo {filename} no encontrado en {path}.")
            except Exception as e:
                print(f"ERROR: No se pudo procesar el archivo {filename} en {path}. Error: {e}")

        # Si encontramos todos los datos, salimos del bucle de rutas
        if len(found_estaciones) == len(estaciones):
            break
        
    if df_all.empty:
        print("¡ERROR CRÍTICO! No se pudo cargar ningún dato.")
        return pd.DataFrame(), []
    
    print(f"Carga global finalizada. Estaciones cargadas: {found_estaciones}")
    return df_all, found_estaciones

df_global, found_estaciones_global = load_data_globally()

if df_global.empty:
    # CORRECCIÓN DE SINTAXIS APLICADA AQUÍ (for e e in estaciones -> for e in estaciones)
    error_message = html.Div(
        [
            html.H1("⚠️ Error Crítico de Carga de Datos", style={'color': '#e74c3c'}),
            html.P("La aplicación no pudo encontrar o procesar ningún archivo CSV de datos."),
            html.P(f"Por favor, asegúrate de que los archivos {', '.join([f'{e}_et0_variants_ajustado.csv' for e in estaciones])} "
                   f"estén en la carpeta 'datos_siar_baleares' dentro de la carpeta del proyecto."),
        ], style={'textAlign': 'center', 'marginTop': '50px'}
    )
    app.layout = dbc.Container(error_message)
else:
    # Obtener el rango de años para el slider
    min_year = df_global['Fecha'].dt.year.min()
    max_year = df_global['Fecha'].dt.year.max()
    years = list(range(min_year, max_year + 1))
    
    # Lista de Modelos (Columnas para análisis)
    modelos = [
        {'label': 'Penman-Monteith (SIAR)', 'value': 'ET0_calc'},
        {'label': 'Hargreaves Ajustado', 'value': 'ET0_harg_ajustado'},
        {'label': 'Valiantzas Ajustado', 'value': 'ET0_val_ajustado'},
    ]
    
    # Columnas para la tabla de errores
    errors_columns = [
        {"name": "Modelo", "id": "Modelo"},
        {"name": "MSE (mm²/d²)", "id": "MSE"},
        {"name": "RRMSE", "id": "RRMSE"},
        {"name": "MAE (mm/día)", "id": "MAE"},
        {"name": "R²", "id": "R2"},
        {"name": "AARE", "id": "AARE"},
    ]
    
    # =========================================================================
    # 2. FUNCIONES DE CÁLCULO DE MÉTRICAS Y GRÁFICOS
    # =========================================================================

    def calculate_errors_df(df_filtered):
        """Calcula las métricas de error agregadas para el DataFrame filtrado."""
        if df_filtered.empty:
            return pd.DataFrame()

        y_true = df_filtered['ET0_calc'].dropna()
        if y_true.empty:
            return pd.DataFrame()
            
        mean_y_true = y_true.mean()
        
        results = []
        for model in modelos[1:]: # Excluir ET0_calc (la referencia)
            col = model['value']
            
            # Asegurar que ambos tienen valores
            comparison_df = df_filtered[['ET0_calc', col]].dropna()
            
            if comparison_df.empty:
                continue

            y_true_comp = comparison_df['ET0_calc']
            y_pred = comparison_df[col]
            
            # Métricas
            mse = ((y_true_comp - y_pred) ** 2).mean()
            rmse = np.sqrt(mse)
            mae = (np.abs(y_true_comp - y_pred)).mean()
            r2 = 1 - (np.sum((y_true_comp - y_pred)**2) / np.sum((y_true_comp - mean_y_true)**2)) if mean_y_true != 0 else 0
            # Evitar división por cero o NaNs en AARE
            aare = (np.abs((y_true_comp - y_pred) / y_true_comp)).replace([np.inf, -np.inf], np.nan).dropna().mean()
            
            results.append({
                "Modelo": model['label'],
                "MSE": f"{mse:.3f}",
                "RRMSE": f"{rmse / mean_y_true:.3f}" if mean_y_true else "N/A",
                "MAE": f"{mae:.3f}",
                "R2": f"{r2:.3f}",
                "AARE": f"{aare:.3f}",
            })
            
        return pd.DataFrame(results)
        
    def create_time_series_fig(df_filtered, selected_models):
        """Crea el gráfico de series temporales para los modelos seleccionados."""
        if df_filtered.empty or not selected_models:
            return go.Figure()

        df_monthly = df_filtered.set_index('Fecha')[selected_models].resample('M').mean().reset_index()
        
        fig = px.line(df_monthly, x='Fecha', y=selected_models, 
                      title='Serie Temporal Mensual de ET₀ (Media)',
                      labels={'value': 'ET₀ (mm/día)', 'Fecha': 'Fecha', 'variable': 'Modelo'},
                      color_discrete_map={
                          'ET0_calc': '#1f77b4',       # Azul PM
                          'ET0_harg_ajustado': '#ff7f0e', # Naranja Hargreaves
                          'ET0_val_ajustado': '#2ca02c'  # Verde Valiantzas
                      })
                      
        fig.update_layout(title_font_size=18, font=font_style, hovermode="x unified")
        return fig

    def create_scatter_diff(df_filtered, x_var, title):
        """Crea un gráfico de dispersión de diferencia vs una variable (Temp/Rs)."""
        if df_filtered.empty:
            return None
            
        df_plot = df_filtered.copy().dropna(subset=['ET0_calc', 'ET0_harg_ajustado', 'ET0_val_ajustado', x_var])
        if df_plot.empty:
            return None

        # Calcular las diferencias
        df_plot['Diff_Harg'] = df_plot['ET0_harg_ajustado'] - df_plot['ET0_calc']
        df_plot['Diff_Val'] = df_plot['ET0_val_ajustado'] - df_plot['ET0_calc']
        
        # Melt para Plotly
        df_melted = df_plot.melt(
            id_vars=[x_var], 
            value_vars=['Diff_Harg', 'Diff_Val'], 
            var_name='Modelo', 
            value_name='Diferencia (mm/día)'
        )
        
        fig = px.scatter(df_melted, x=x_var, y='Diferencia (mm/día)', color='Modelo', 
                         title=title,
                         labels={x_var: x_var, 'Diferencia (mm/día)': 'Error de Estimación'},
                         template="plotly_white")
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Diferencia Cero")
        fig.update_traces(marker=dict(size=3, opacity=0.5))
        fig.update_layout(title_font_size=18, font=font_style, legend_title_text="Modelo")
        return fig

    def create_monthly_diff_fig(df_filtered):
        """Crea el gráfico de barras de diferencias medias mensuales."""
        if df_filtered.empty:
            return None
            
        df_plot = df_filtered.copy().dropna(subset=['ET0_calc', 'ET0_harg_ajustado', 'ET0_val_ajustado'])
        if df_plot.empty:
            return None

        # Calcular diferencias y mes
        df_plot['Diff_Harg'] = df_plot['ET0_harg_ajustado'] - df_plot['ET0_calc']
        df_plot['Diff_Val'] = df_plot['ET0_val_ajustado'] - df_plot['ET0_calc']
        df_plot['Mes'] = df_plot['Fecha'].dt.month
        
        # Calcular media mensual de las diferencias
        df_monthly_diff = df_plot.groupby('Mes')[['Diff_Harg', 'Diff_Val']].mean().reset_index()
        
        # Melt para Plotly
        df_melted = df_monthly_diff.melt(
            id_vars=['Mes'], 
            value_vars=['Diff_Harg', 'Diff_Val'], 
            var_name='Modelo', 
            value_name='Diferencia Media Mensual (mm/día)'
        )
        
        # Mapear números de mes a nombres
        month_names = {1:'Ene', 2:'Feb', 3:'Mar', 4:'Abr', 5:'May', 6:'Jun', 7:'Jul', 8:'Ago', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dic'}
        df_melted['Mes_Nombre'] = df_melted['Mes'].map(month_names)

        fig = px.bar(df_melted, x='Mes_Nombre', y='Diferencia Media Mensual (mm/día)', color='Modelo', 
                     barmode='group', 
                     title='Sesgo Promedio Mensual (Modelo - PM)',
                     labels={'Mes_Nombre': 'Mes', 'Diferencia Media Mensual (mm/día)': 'Error (mm/día)'},
                     template="plotly_white")

        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Diferencia Cero")
        fig.update_layout(title_font_size=18, font=font_style, legend_title_text="Modelo")
        return fig


    # =========================================================================
    # 3. DISEÑO DE LA APLICACIÓN DASH
    # =========================================================================

    app.layout = dbc.Container([
        dbc.Row(dbc.Col(html.H1("Dashboard de Análisis de ET₀ - Islas Baleares", className="text-center my-4", style={'color': '#1abc9c', 'fontFamily': 'Arial, sans-serif'}))),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Label("Seleccionar Estación:", style=header_style),
                    dcc.Dropdown(
                        id='dropdown-estacion',
                        options=[{'label': f'Estación {e}', 'value': e} for e in found_estaciones_global],
                        value=found_estaciones_global[0] if found_estaciones_global else None,
                        clearable=False,
                        style=font_style
                    ),
                ], className="mb-4"),
                
                html.Div([
                    html.Label("Seleccionar Rango de Años:", style=header_style),
                    dcc.RangeSlider(
                        id='slider-year',
                        min=min_year,
                        max=max_year,
                        value=[min_year, max_year],
                        marks={str(y): str(y) for y in years if y % 5 == 0 or y == min_year or y == max_year},
                        step=1,
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ], className="mb-4 pt-3"),
                
                html.Div([
                    html.Label("Modelos a Visualizar en Serie Temporal:", style=header_style),
                    dcc.Checklist(
                        id='checklist-modelos',
                        options=modelos,
                        value=[m['value'] for m in modelos],
                        inline=False,
                        style=font_style,
                        labelStyle={'display': 'block', 'cursor': 'pointer'}
                    ),
                ], className="mb-4"),
            ], md=4, className="bg-light p-4 rounded-3 shadow-lg"),
            
            dbc.Col(html.Div(id='output-dashboard'), md=8),
        ], className="my-4"),
        
        html.Div(id='data-error-message', style={'color': '#e74c3c', 'fontWeight': 'bold', 'textAlign': 'center', 'marginTop': '20px'}),

    ], fluid=True, style={'backgroundColor': '#ecf0f1', 'minHeight': '100vh'})


    # =========================================================================
    # 4. CALLBACKS
    # =========================================================================

    @app.callback(
        [Output('output-dashboard', 'children'),
         Output('data-error-message', 'children')],
        [Input('dropdown-estacion', 'value'),
         Input('slider-year', 'value'),
         Input('checklist-modelos', 'value')]
    )
    def update_dashboard(code, year_range, selected_models):
        error_msg = ""
        
        # 1. Filtrar los datos
        try:
            # Asegurarse de que el código exista antes de filtrar
            if code not in found_estaciones_global:
                error_msg = f"🚨 La estación seleccionada ({code}) no tiene datos cargados."
                return html.Div(error_msg), error_msg
                
            df_filtered = df_global[df_global['Estacion'] == code].copy()

            # Filtrar por rango de años
            start_year, end_year = year_range
            df_filtered = df_filtered[
                (df_filtered['Fecha'].dt.year >= start_year) & 
                (df_filtered['Fecha'].dt.year <= end_year)
            ]
            
            if df_filtered.empty:
                error_msg = f"🚨 No hay datos disponibles para la estación {code} en el rango {start_year}-{end_year}."
                return html.Div(error_msg), error_msg

            # 2. Generar Componentes
            
            # Cálculo de Errores
            errors_df = calculate_errors_df(df_filtered)
            
            # Gráficos
            fig_time = create_time_series_fig(df_filtered, selected_models)
            fig_diff_temp = create_scatter_diff(df_filtered, 'TempMedia', 'Diferencia vs Temperatura Media (PM vs Empíricos)')
            fig_diff_rs = create_scatter_diff(df_filtered, 'Radiacion', 'Diferencia vs Radiación (PM vs Empíricos)')
            fig_diff_month = create_monthly_diff_fig(df_filtered)
            
            # 3. Construir el Layout de Salida
            content = [
                html.H2(f"Análisis de ET₀ para Estación {code}", style={'color': '#2c3e50', 'marginTop': '0'}),
                
                dbc.Card([
                    dbc.CardHeader("Métricas de Error (vs Penman-Monteith)", style={'backgroundColor': '#2c3e50', 'color': 'white'}),
                    dbc.CardBody(
                         dash_table.DataTable(
                            id='error-table',
                            data=errors_df.to_dict('records'),
                            columns=errors_columns,
                            style_cell={**table_style, 'border': '1px solid #ddd'},
                            style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa', 'color': '#2c3e50'},
                            style_table={'overflowX': 'auto', 'marginBottom': '0px', 'borderRadius': '0 0 0.25rem 0.25rem'},
                        ),
                    ),
                ], className="mb-4 shadow-sm"),
                
                dbc.Card([
                    dbc.CardHeader("Serie Temporal de ET₀"),
                    dbc.CardBody(dcc.Graph(figure=fig_time)),
                ], className="mb-4 shadow-sm"),
                
                dbc.Card([
                    dbc.CardHeader("Sesgo Promedio Mensual (Modelo - PM)"),
                    dbc.CardBody(dcc.Graph(figure=fig_diff_month) if fig_diff_month else html.P('Datos insuficientes para el gráfico.', style=font_style)),
                ], className="mb-4 shadow-sm"),

                dbc.CardGroup([
                    dbc.Card([
                        dbc.CardHeader("Diferencias vs Temperatura Media"),
                        dbc.CardBody(dcc.Graph(figure=fig_diff_temp) if fig_diff_temp else html.P('Datos insuficientes.', style=font_style)),
                    ], className="shadow-sm"),
                    dbc.Card([
                        dbc.CardHeader("Diferencias vs Radiación"),
                        dbc.CardBody(dcc.Graph(figure=fig_diff_rs) if fig_diff_rs else html.P('Datos insuficientes.', style=font_style)),
                    ], className="shadow-sm"),
                ], className="mb-4"),
            ]
            return content, error_msg
        
        except Exception as e:
            error_msg = f"🚨 Error crítico procesando la estación {code}: {str(e)}"
            print(error_msg)
            # Retorna una vista de error amigable en la app
            return html.Div([
                html.H3("¡Error Inesperado en el Dashboard!", style={'color': '#e74c3c'}),
                html.P(f"El procesamiento falló. Detalle: {str(e)}"),
            ]), error_msg

if __name__ == '__main__':
    print("Dash app está lista para ser servida por Gunicorn o similar.")
    # La versión de Dash instalada usa app.run() en lugar de app.run_server()
    print("Iniciando servidor de desarrollo local de Dash...")
    app.run(debug=True)