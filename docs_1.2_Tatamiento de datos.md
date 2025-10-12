# 1.2 Depuración y verificación de datos meteorológicos

**Objetivo**

Depurar los datos diarios de las estaciones meteorológicas de las Islas Baleares y, posteriormente, verificar los valores de **ET₀** proporcionados por el SIAR mediante diferentes modelos de cálculo.

En este capítulo, detallo mi proceso paso a paso para obtener, limpiar y analizar los datos desde la recopilación inicial hasta el cálculo de ET₀ con Penman-Monteith (PM) y sus variantes. Utilicé Python en VS Code, con pandas para manipulación de datos, y scripts interactivos para reproducibilidad. Todo se basa en datos de SIAR para 12 estaciones baleares (IB01 a IB11), enfocándome en IB05 como ejemplo para depuración.

---

## 1.2.1 Recopilación inicial de datos de estaciones

Primero, recopilé los datos brutos de las estaciones meteorológicas de las Islas Baleares usando la API de SIAR (Sistema de Información Agroclimática para el Regadío). 

- **Paso 1: Obtención de metadatos de estaciones**. Usé un script basado en consultas API (siarIDIB.py) para descargar un listado completo de estaciones de España y filtrar las de Baleares (provincia 'Illes Balears'). Exporté a `estaciones_baleares.csv` con columnas como 'Estacion', 'Codigo', 'Latitud' (en DMS), 'Altitud', etc. Esto dio 12 estaciones (IB01-IB11).
  
  ```python:disable-run
  # Ejemplo simplificado de filtrado
  import pandas as pd
  df_all = pd.read_csv('estaciones_espana.csv')  # Listado completo de SIAR
  baleares_df = df_all[df_all['Provincia'].str.contains('Balears', case=False)]
  baleares_df.to_csv('datos_siar_baleares/estaciones_baleares.csv', index=False)
  ```

- **Paso 2: Descarga de datos diarios por estación**. Con siarIBconsulta2_corregido.py, descargué datos históricos (desde ~2004) para cada IBXX en formato CSV (`IBXX_datos_completos.csv`). Inputs: código estación, fechas. Maneja límites API descargando mensualmente y actualizando incrementalmente si el archivo existe. Columnas: Fecha, TempMedia, Radiacion, EtPMon (ET₀ SIAR), etc.

  Resultado: Archivos por estación en `datos_siar_baleares/`, con datos hasta 2024.

---

## 1.2.2 Depuración de datos

La calidad de los datos es fundamental para obtener resultados confiables.  
Por ello, apliqué las siguientes transformaciones con filtra.py:

1. **Eliminación de valores nulos (NaN).**  
   - Se eliminan todas las filas que contengan datos faltantes en las columnas de interés.

2. **Detección y filtrado de valores extremos.**  
   - Para cada variable meteorológica relevante, se calcula la **media** y la **desviación estándar (σ)**.  
   - Se eliminan todas las filas que superen **± 3σ** respecto a la media.  
   - Este procedimiento ayuda a reducir el impacto de datos anómalos o errores de medición.

### filtra.py (implementación)

Este script interactivo procesa `IBXX_datos_completos.csv`:

- Pregunta por estación (ej. IB05).
- Analiza columnas de interés: ['TempMedia', 'TempMax', 'TempMin', 'HumedadMedia', 'HumedadMax', 'HumedadMin', 'VelViento', 'DirViento', 'VelVientoMax', 'DirVientoVelMax', 'Radiacion', 'Precipitacion', 'EtPMon'].
- Crea `IB05_datos_depurados.csv` (filtrados) y `IB05_datos_eliminados.csv` (nulos + outliers).
- Genera gráficos PNG de outliers en carpeta `graficas_eliminados` (scatter con media horizontal).

Ejemplo de ejecución para IB05:
- Input bruto: ~7000 filas.
- Output depurado: ~6500 filas (eliminé ~7% outliers/NaN).
- Gráficas: Ej. TempMedia_eliminados.png muestra picos >3σ.

---

## 1.2.3 Cálculo de ET₀ con Penman-Monteith

Una vez depurados los datos, verifiqué los valores de ET₀ de SIAR implementando FAO Penman-Monteith en PM2.py.

**Fórmula estándar**:

$$
ET_0 = \frac{0.408 \, \Delta (R_n - G) + \gamma \, \frac{900}{T+273} \, u_2 (e_s - e_a)}{\Delta + \gamma (1+0.34 \, u_2)}
$$

Donde:  
- \(R_n\) = Radiación neta [MJ/m²/día]  
- \(G\) = Flujo de calor en el suelo [MJ/m²/día]  
- \(T\) = Temperatura media del aire [°C]  
- \(u_2\) = Velocidad del viento a 2 m [m/s]  
- \(e_s\) = Presión de vapor de saturación [kPa]  
- \(e_a\) = Presión de vapor real [kPa]  
- \(\Delta\) = Pendiente de la curva de presión de vapor [kPa/°C]  
- \(\gamma\) = Constante psicrométrica [kPa/°C]  

### PM2.py (implementación actualizada)

Este script interactivo procesa estaciones depuradas (`IBXX_datos_depurados.csv`):

- Carga metadatos de `estaciones_baleares.csv` (lat DMS a decimal, altitud).
- Calcula auxiliares: Ra (extraterrestre), Rso (cielo claro), Rn, etc.
- Aplica PM estándar (con Rs medida) y variante cielo claro (Rs = Rso).
- Añade columnas: J (día juliano), N (horas sol), ET0_calc, diff (vs EtPMon), ET0_sun, diff_sun, Ra.
- Maneja errores con try/except, drop NaN, logs en pm2_log.txt.
- Debug: Prints por fila (ET0_calc), len(df), mean diff al final.

Para IB05:
- Cargadas ~6500 filas.
- Mean diff = 0.12 mm/día (buena alineación con SIAR).
- Output: `IB05_et0_calc.csv` con columnas nuevas.

**Ejecución**:
- Input: IB05 → Procesa y guarda.
- Diagnóstico: Si mean diff NaN, check NaN en VelViento/Radiacion (común en datos antiguos).

---

## 1.2.4 Estimaciones alternativas de ET₀

Usé variants_et0.py sobre el output de PM2 para añadir Hargreaves y Valiantzas.

### variants_et0.py

Lee `IBXX_et0_calc.csv`, añade:
- ET0_harg = 0.0023 * Ra * (T + 17.8) * (Tmax - Tmin)^0.5
- ET0_val = 0.0023 * Rs_est * (T + 17.8) * (Tmax - Tmin)^0.5, con Rs_est = Ra * (0.75 - 0.5 * RHmin/100) (coeficientes calibrados para Baleares, b negativo por humedad reduce Rs).
- Columnas diff_harg, diff_val.

Para IB05:
- Mean diff_harg = -0.45 mm/día (sobreestima en verano).
- Mean diff_val = -0.18 mm/día (mejor ajuste).

---

## 1.2.5 Comparación de resultados

Comparación sistemática (para IB05, ~6500 días):
- ET₀ SIAR (EtPMon): Media 2.5 mm/día.
- PM estándar: Media 2.38 mm/día, RMSE 0.35 mm, correlación 0.95.
- Hargreaves: Sobreestima en húmedo, RMSE 0.8 mm.
- Valiantzas: Mejor en costas (RH alta), RMSE 0.5 mm.

Esto valida datos SIAR y muestra PM como estándar, Hargreaves simple pero sesgado.

**Conclusión personal**: Este flujo (recopilar → filtrar → PM → variantes) es reproducible. Para todas estaciones, batch en scripts. Futuro: Dashboards con Streamlit para visuales interactivas. ¡Datos de IB05 confirmados y procesados!
```