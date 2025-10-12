import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.express as px

# Path to data
data_path = r'C:/Users/josep/OneDrive/Documentos/GitHub/Capitulo-1/datos_siar_baleares'

# Lista estaciones disponibles (IB01 a IB11, ajusta si más)
estaciones = [f'IB{str(i).zfill(2)}' for i in range(1, 12)]

# Título
st.title('Análisis Comparativo de Modelos ET₀ por Estación')

# Desplegable para seleccionar estación
code = st.selectbox('Selecciona estación', estaciones)

# Cargar datos
file = os.path.join(data_path, f'{code}_et0_variants.csv')
if not os.path.exists(file):
    st.error(f"Archivo {file} no encontrado. Ejecuta variants_et0.py primero.")
else:
    df = pd.read_csv(file, parse_dates=['Fecha'])
    df = df.dropna(subset=['EtPMon', 'ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val'])  # Limpia NaN
    
    st.write(f"Datos para {code}: {len(df)} filas. Mean ET₀ SIAR: {df['EtPMon'].mean():.2f} mm/día")
    
    # Serie temporal
    st.subheader('Serie Temporal de ET₀')
    fig_time = px.line(df, x='Fecha', y=['EtPMon', 'ET0_calc', 'ET0_sun', 'ET0_harg', 'ET0_val'],
                       title=f'Comparación Temporal - {code}',
                       labels={'value': 'ET₀ (mm/día)', 'variable': 'Modelo'})
    st.plotly_chart(fig_time)
    
    # Scatter ET0_calc vs SIAR
    st.subheader('Scatter PM vs SIAR')
    fig_scatter = px.scatter(df, x='EtPMon', y='ET0_calc', trendline='ols',
                             title=f'ET0_calc vs EtPMon - {code}',
                             labels={'EtPMon': 'ET₀ SIAR', 'ET0_calc': 'ET0 PM'})
    st.plotly_chart(fig_scatter)
    
    # Histograma diferencias
    st.subheader('Histograma de Diferencias (SIAR - Modelos)')
    df_diff = df[['diff', 'diff_sun', 'diff_harg', 'diff_val']].melt(var_name='Modelo', value_name='Diferencia')
    fig_hist = px.histogram(df_diff, x='Diferencia', color='Modelo', nbins=50,
                            title='Distribución Diferencias')
    st.plotly_chart(fig_hist)
    
    # Métricas
    st.subheader('Métricas de Error')
    metrics = pd.DataFrame({
        'Modelo': ['PM', 'PM_sun', 'Harg', 'Val'],
        'RMSE': [
            ((df['diff'])**2).mean()**0.5,
            ((df['diff_sun'])**2).mean()**0.5,
            ((df['diff_harg'])**2).mean()**0.5,
            ((df['diff_val'])**2).mean()**0.5
        ]
    })
    st.table(metrics)