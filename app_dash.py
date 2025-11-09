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
# 1. DEFINICIÓN DE MODELOS Y DESCRIPCIONES (Añadido)
# =========================================================================

# Diccionario con las descripciones detalladas de los modelos para el UI
model_descriptions = {
    'EtPMon': {
        'nombre': 'Penman-Monteith FAO-56 (SIAR Referencia)',
        'descripcion': 'El modelo de referencia (Gold Standard) utilizado por el SIAR para calcular la evapotranspiración de referencia (ET₀). Es la base de comparación para evaluar la precisión de todos los demás modelos empíricos y de Machine Learning.',
        'variables': 'Temperaturas (Min, Max), Humedad, Viento, Radiación y presión atmosférica.'
    },
    'ET0_calc': {
        'nombre': 'Penman-Monteith Estándar (Calculado)',
        'descripcion': 'Implementación del modelo Penman-Monteith según la metodología FAO-56, utilizando todas las variables disponibles. Se incluye como una verificación directa del cálculo frente a la referencia del SIAR (EtPMon).',
        'variables': 'Temperaturas (Min, Max), Humedad, Viento, Radiación y Ra.'
    },
    'ET0_harg_ajustado': {
        'nombre': 'Hargreaves Ajustado (HGR ajustado)',
        'descripcion': 'Modelo empírico que utiliza la temperatura y la radiación extraterrestre (Ra). Es simple e ideal si faltan datos de viento/humedad. La versión **ajustada** utiliza un Coeficiente de Ajuste (AHC) calibrado para mejorar su desempeño en Baleares.',
        'variables': 'Temperaturas (Min, Max) y Radiación extraterrestre (Ra).'
    },
    'ET0_val_ajustado': {
        'nombre': 'Valiantzas Ajustado (VAL ajustado)',
        'descripcion': 'Modelo empírico que combina temperatura, radiación solar (Rs) y humedad, requiriendo más datos que Hargreaves. La versión **ajustada** usa un coeficiente calibrado, destacando en el análisis de errores como el mejor modelo empírico de las variantes.',
        'variables': 'Temperaturas (Min, Max), Radiación Solar (Rs) y Humedad Media.'
    }
}

# =========================================================================
# 2. CARGA GLOBAL DE DATOS
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
    
    # Intenta cargar los datos de las estaciones definidas
    for code in estaciones:
        file_path = os.path.join(data_path, f'{code}_et0_variants_ajustado.csv')
        try:
            # La carga global se hace más robusta para el entorno Render
            df = pd.read_csv(file_path, parse_dates=['Fecha'])
            # Asegurar que las columnas de diferencia necesarias existan para el análisis ajustado
            if 'diff' not in df.columns:
                df['diff'] = df['ET0_calc'] - df['EtPMon']
            if 'diff_harg_ajustado' not in df.columns:
                df['diff_harg_ajustado'] = df['ET0_harg_ajustado'] - df['EtPMon']
            if 'diff_val_ajustado' not in df.columns:
                df['diff_val_ajustado'] = df['ET0_val_ajustado'] - df['EtPMon']
                
            df_all = pd.concat([df_all, df], ignore_index=True)
            found_estaciones.append(code)
            print(f"✅ Cargado {code} con {len(df)} filas.")
            
        except FileNotFoundError:
            print(f"⚠️ Archivo {file_path} no encontrado. Omitiendo estación {code}.")
        except Exception as e:
            print(f"🚨 Error leyendo {file_path}: {e}")

    if df_all.empty:
        print("❌ No se pudo cargar ningún archivo de datos. La aplicación funcionará sin datos.")
        return pd.DataFrame(), []

    # Cálculos globales para la tabla de errores
    if not df_all.empty:
        # Calcular los errores promedio globales (absoluto y relativo)
        errors = []
        modelos_analisis = ['ET0_calc', 'ET0_harg_ajustado', 'ET0_val_ajustado']
        for model_col in modelos_analisis:
            df_temp = df_all.dropna(subset=[model_col, 'EtPMon'])
            
            # Las columnas de diferencia se llaman 'diff', 'diff_harg_ajustado', 'diff_val_ajustado'
            # y ya deben estar calculadas arriba o en el CSV.
            diff_col = f'diff' if model_col == 'ET0_calc' else f'diff_{model_col.split("_")[1]}_ajustado'
            
            # Reutilizando el cálculo de errores de docs_1.3_Análisis errores.md
            if diff_col in df_temp.columns:
                diff = df_temp[diff_col]
                mse = np.mean(diff ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(diff))
                
                # RRMSE y AARE requieren el valor medio de referencia (EtPMon)
                etpm_mean = df_temp['EtPMon'].mean()
                rrmse = (rmse / etpm_mean) * 100 if etpm_mean != 0 else np.nan
                aare = np.mean(np.abs(diff / df_temp['EtPMon'])) * 100 if etpm_mean != 0 else np.nan
                
                # R2 Score
                ss_res = np.sum(diff ** 2)
                ss_tot = np.sum((df_temp['EtPMon'] - etpm_mean) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

                errors.append({
                    'Modelo': model_descriptions[model_col]['nombre'],
                    'MSE (mm²/día²)': f'{mse:.4f}',
                    'RRMSE (%)': f'{rrmse:.2f}',
                    'MAE (mm/día)': f'{mae:.3f}',
                    'R²': f'{r2:.4f}',
                    'AARE (%)': f'{aare:.2f}'
                })
        
        df_errors_global = pd.DataFrame(errors)
    else:
        df_errors_global = pd.DataFrame()
    
    return df_all, found_estaciones, df_errors_global

# Ejecutar la carga de datos una sola vez al inicio
df_data, available_estaciones, df_errors_global = load_data_globally()

# =========================================================================
# 3. LAYOUT DE LA APLICACIÓN
# =========================================================================

app.layout = dbc.Container([
    html.H1("💧 Análisis de Modelos de Evapotranspiración (ET₀) en Baleares", className="text-center my-4", style={'color': '#2c3e50'}),
    
    dbc.Alert(
        [
            html.H5("Análisis Global de Errores (Todas las Estaciones)", className="alert-heading"),
            "Este resumen muestra el rendimiento promedio de los modelos en todas las estaciones disponibles. Valiantzas Ajustado es el mejor modelo empírico.",
            html.Hr(),
            dash_table.DataTable(
                id='global-error-table',
                data=df_errors_global.to_dict('records'),
                columns=[{"name": i, "id": i} for i in df_errors_global.columns],
                style_cell=table_style,
                style_header={'fontWeight': 'bold', 'backgroundColor': '#cce5ff', 'color': '#004085'},
                style_data_conditional=[
                    {
                        'if': {'column_id': 'Modelo', 'filter_query': '{Modelo} contains "Penman-Monteith"'},
                        'backgroundColor': '#f8d7da',
                        'color': '#721c24'
                    }
                ],
            )
        ],
        color="info",
        className="my-4"
    ),

    dbc.Row([
        dbc.Col(html.Div([
            html.H3("Seleccionar Estación", style=header_style),
            dcc.Dropdown(
                id='estacion-dropdown',
                options=[{'label': f'Estación {code}', 'value': code} for code in available_estaciones],
                value=available_estaciones[0] if available_estaciones else None,
                clearable=False,
                style=font_style
            ),
        ]), md=4),
        dbc.Col(html.Div(id='error-message'), md=8, style={'align-self': 'center'}),
    ]),

    html.Hr(className="my-4"),

    # Contenido dinámico (descripciones, tablas y gráficos)
    html.Div(id='output-content')

], fluid=True, style={'maxWidth': '1200px', 'backgroundColor': '#f9f9f9', 'padding': '20px', 'borderRadius': '8px'})

# =========================================================================
# 4. CALLBACKS DE INTERACCIÓN
# =========================================================================

@app.callback(
    [Output('output-content', 'children'),
     Output('error-message', 'children')],
    [Input('estacion-dropdown', 'value')]
)
def update_output(code):
    error_msg = html.P('', style={'color': 'red'})
    
    if not code:
        return html.Div(), html.P('Seleccione una estación para comenzar el análisis.', style={'color': '#007bff'})
    
    # Filtrar datos por estación
    df_station = df_data[df_data['Estacion'] == code].copy()
    
    if df_station.empty:
        error_msg = html.P(f'🚨 No hay datos cargados para la estación {code}.', style={'color': 'red'})
        return html.Div(), error_msg

    try:
        df = df_station.set_index('Fecha').dropna(subset=['EtPMon'])
        df['Mes'] = df.index.month
        df['Año'] = df.index.year

        # =========================================================================
        # 4.1 Definición de Columnas para Análisis (Ajustado a la solicitud del usuario)
        # =========================================================================
        
        # Columnas de ET0 para los gráficos de series temporales
        et0_columns = {
            'EtPMon': model_descriptions['EtPMon']['nombre'],
            'ET0_calc': model_descriptions['ET0_calc']['nombre'],
            'ET0_harg_ajustado': model_descriptions['ET0_harg_ajustado']['nombre'],
            'ET0_val_ajustado': model_descriptions['ET0_val_ajustado']['nombre']
        }
        
        # Columnas de Diferencia para la tabla de errores y gráficos de dispersión
        # Comparación contra la referencia EtPMon
        diff_columns = {
            'diff': f'Error {model_descriptions["ET0_calc"]["nombre"]}', # ET0_calc - EtPMon
            'diff_harg_ajustado': f'Error {model_descriptions["ET0_harg_ajustado"]["nombre"]}', # ET0_harg_ajustado - EtPMon
            'diff_val_ajustado': f'Error {model_descriptions["ET0_val_ajustado"]["nombre"]}'  # ET0_val_ajustado - EtPMon
        }

        # =========================================================================
        # 4.2 Generación de Tablas de Errores por Estación
        # =========================================================================
        
        errors = []
        for model_col, model_name in zip(et0_columns.keys() - ['EtPMon'], diff_columns.values()):
            df_temp = df.dropna(subset=[model_col, 'EtPMon'])
            
            # Obtener la columna de diferencia correspondiente
            if model_col == 'ET0_calc':
                 diff_col = 'diff'
            else:
                 diff_col = f'diff_{model_col.split("_")[1]}_ajustado'
            
            if diff_col in df_temp.columns:
                diff = df_temp[diff_col]
                mse = np.mean(diff ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(diff))
                
                etpm_mean = df_temp['EtPMon'].mean()
                rrmse = (rmse / etpm_mean) * 100 if etpm_mean != 0 else np.nan
                aare = np.mean(np.abs(diff / df_temp['EtPMon'])) * 100 if etpm_mean != 0 else np.nan
                
                ss_res = np.sum(diff ** 2)
                ss_tot = np.sum((df_temp['EtPMon'] - etpm_mean) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

                errors.append({
                    'Modelo': model_name.replace('Error ', ''), # Solo el nombre del modelo
                    'MSE (mm²/día²)': f'{mse:.4f}',
                    'RRMSE (%)': f'{rrmse:.2f}',
                    'MAE (mm/día)': f'{mae:.3f}',
                    'R²': f'{r2:.4f}',
                    'AARE (%)': f'{aare:.2f}'
                })
        
        errors_df = pd.DataFrame(errors)
        errors_columns = [{"name": i, "id": i} for i in errors_df.columns]


        # =========================================================================
        # 4.3 Generación de Gráficos
        # =========================================================================

        # Gráfico 1: Serie Temporal de ET₀
        df_time_plot = df.reset_index()[['Fecha'] + list(et0_columns.keys())].melt(
            id_vars='Fecha', value_vars=et0_columns.keys(), 
            var_name='Modelo_Code', value_name='ET0 (mm/día)'
        )
        df_time_plot['Modelo'] = df_time_plot['Modelo_Code'].map(et0_columns)
        
        fig_time = px.line(df_time_plot, x='Fecha', y='ET0 (mm/día)', color='Modelo', 
                           title=f'Serie Temporal de ET₀ en Estación {code} ({df_time_plot["Fecha"].min().year} - {df_time_plot["Fecha"].max().year})',
                           template="plotly_white")
        fig_time.update_layout(height=500, font=font_style, margin={"t":50, "b":10, "l":10, "r":10})
        fig_time.update_traces(opacity=0.8)

        # Gráfico 2: Diferencias vs Temperatura Media
        fig_diff_temp = None
        if 'TempMedia' in df.columns and len(df) > 100:
            df_diff_temp_plot = df.reset_index()[['TempMedia'] + list(diff_columns.keys())].melt(
                id_vars='TempMedia', value_vars=diff_columns.keys(), 
                var_name='Error_Code', value_name='Diferencia (mm/día)'
            )
            df_diff_temp_plot['Modelo'] = df_diff_temp_plot['Error_Code'].map(diff_columns)

            fig_diff_temp = px.scatter(df_diff_temp_plot, x='TempMedia', y='Diferencia (mm/día)', color='Modelo',
                                       title=f'Error de ET₀ vs Temperatura Media (°C)',
                                       template="plotly_white", opacity=0.6)
            fig_diff_temp.add_hline(y=0, line_dash="dash", line_color="red")
            fig_diff_temp.update_layout(height=500, font=font_style, margin={"t":50, "b":10, "l":10, "r":10})

        # Gráfico 3: Diferencias vs Radiación Solar (Rs)
        fig_diff_rs = None
        if 'Radiacion' in df.columns and len(df) > 100:
            df_diff_rs_plot = df.reset_index()[['Radiacion'] + list(diff_columns.keys())].melt(
                id_vars='Radiacion', value_vars=diff_columns.keys(), 
                var_name='Error_Code', value_name='Diferencia (mm/día)'
            )
            df_diff_rs_plot['Modelo'] = df_diff_rs_plot['Error_Code'].map(diff_columns)

            fig_diff_rs = px.scatter(df_diff_rs_plot, x='Radiacion', y='Diferencia (mm/día)', color='Modelo',
                                     title=f'Error de ET₀ vs Radiación Solar ($MJ/m^2$)',
                                     template="plotly_white", opacity=0.6)
            fig_diff_rs.add_hline(y=0, line_dash="dash", line_color="red")
            fig_diff_rs.update_layout(height=500, font=font_style, margin={"t":50, "b":10, "l":10, "r":10})

        # Gráfico 4: Diferencias Mensuales (MAE)
        fig_diff_month = None
        if len(df) > 100:
            df_monthly_diff = df.reset_index()[['Mes'] + list(diff_columns.keys())].melt(
                id_vars='Mes', value_vars=diff_columns.keys(), 
                var_name='Error_Code', value_name='Diferencia (mm/día)'
            )
            df_monthly_diff['Modelo'] = df_monthly_diff['Error_Code'].map(diff_columns)
            
            df_monthly_mae = df_monthly_diff.groupby(['Mes', 'Modelo'])['Diferencia (mm/día)'].apply(lambda x: np.mean(np.abs(x))).reset_index(name='MAE Mensual')
            
            # Mapear números de mes a nombres para mejor visualización
            month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
            df_monthly_mae['Mes_Nombre'] = df_monthly_mae['Mes'].apply(lambda x: month_names[x-1])

            fig_diff_month = px.bar(df_monthly_mae, x='Mes_Nombre', y='MAE Mensual', color='Modelo', barmode='group',
                                    title=f'Error Absoluto Medio (MAE) Mensual',
                                    template="plotly_white")
            fig_diff_month.update_layout(height=500, font=font_style, margin={"t":50, "b":10, "l":10, "r":10}, xaxis={'categoryorder':'array', 'categoryarray':month_names})
        
        # =========================================================================
        # 4.4 Contenido HTML del Output (Añadidas descripciones)
        # =========================================================================

        # Función auxiliar para renderizar descripciones
        def render_descriptions(models):
            items = []
            for code in models:
                desc = models[code]
                items.append(
                    dbc.Card(
                        [
                            dbc.CardHeader(html.H5(desc['nombre'], className="mb-0", style={'color': '#2c3e50'})),
                            dbc.CardBody([
                                html.P(desc['descripcion'], style=font_style),
                                html.P(f"Variables de Entrada: {desc['variables']}", className="fst-italic small", style={'color': '#555'})
                            ])
                        ],
                        className="mb-3 shadow-sm",
                    )
                )
            return items


        content = [
            html.H2(f"Detalle de Análisis para la Estación {code}", className="text-center my-4", style={'color': '#007bff'}),
            
            html.H3('1. Descripción de Modelos de Estimación', style={'fontWeight': 'bold', 'color': '#2c3e50', 'marginTop': '20px'}),
            dbc.Row([
                dbc.Col(render_descriptions({k: model_descriptions[k] for k in ['EtPMon', 'ET0_calc']}), md=6),
                dbc.Col(render_descriptions({k: model_descriptions[k] for k in ['ET0_harg_ajustado', 'ET0_val_ajustado']}), md=6),
            ]),
            
            html.H3(f'2. Errores Comparativos de los Modelos (Vs. {model_descriptions["EtPMon"]["nombre"]})', style=header_style),
            html.P(f"La tabla muestra las métricas de error (MAE, RRMSE, R²) para los modelos empíricos en comparación con la referencia del SIAR para la estación {code}.", style=font_style),
            dash_table.DataTable(
                id='station-error-table',
                data=errors_df.to_dict('records'),
                columns=errors_columns,
                style_cell=table_style,
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                style_table={'marginBottom': '20px'},
            ),
            
            html.H3('3. Serie Temporal de ET₀', style=header_style),
            html.P("Comparación directa de los valores de ET₀ calculados por los modelos frente a la referencia a lo largo del tiempo. Las diferencias son más evidentes en los meses de verano.", style=font_style),
            dcc.Graph(figure=fig_time),
            
            html.H3('4. Diferencias vs Temperatura Media', style=header_style),
            html.P("Gráfico de dispersión que muestra cómo varía el error (la diferencia entre el modelo y la referencia) en función de la temperatura. Una dispersión cercana a la línea roja (y=0) indica alta precisión. Se observa la tendencia a sobreestimar (puntos por encima de 0) en los modelos empíricos a ciertas temperaturas.", style=font_style),
            dcc.Graph(figure=fig_diff_temp) if fig_diff_temp else html.P('Datos insuficientes para el gráfico.', style=font_style),
            
            html.H3('5. Diferencias vs Radiación', style=header_style),
            html.P("Análisis de cómo la radiación solar afecta la precisión de cada modelo, especialmente los basados en Rs (Valiantzas) o Ra (Hargreaves).", style=font_style),
            dcc.Graph(figure=fig_diff_rs) if fig_diff_rs else html.P('Datos insuficientes para el gráfico.', style=font_style),
            
            html.H3('6. Error Absoluto Medio (MAE) Mensual', style=header_style),
            html.P("Visualización del error (MAE) agrupado por mes. Los errores suelen aumentar en verano debido a la mayor variabilidad climática y los valores absolutos más altos de ET₀.", style=font_style),
            dcc.Graph(figure=fig_diff_month) if fig_diff_month else html.P('Datos insuficientes para el gráfico.', style=font_style),
        ]
        return content, error_msg
    
    except Exception as e:
        error_msg = html.P(f"🚨 Error crítico procesando la estación {code}: {str(e)}", style={'color': 'red'})
        print(f"Error en update_output para {code}: {str(e)}")
        return html.Div(), error_msg

if __name__ == '__main__':
    # Esta parte no se ejecuta en Render, pero es útil para pruebas locales
    app.run_server(debug=True)