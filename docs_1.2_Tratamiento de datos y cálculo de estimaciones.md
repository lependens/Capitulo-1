# 1.2 Depuración y Verificación de Datos Meteorológicos: Guía Paso a Paso

**Objetivo**

Esta sección describe el proceso completo para depurar y verificar los datos diarios de las estaciones meteorológicas de las Islas Baleares, enfocándonos en la estimación de **ET₀** (evapotranspiración de referencia) proporcionada por el SIAR. El objetivo es limpiar los datos brutos, calcular ET₀ usando el modelo Penman-Monteith (PM) estándar y variantes, agregar estimaciones alternativas (Hargreaves y Valiantzas), y finalmente evaluar los errores comparativos.

Utilicé Python 3 en Visual Studio Code (VS Code), con bibliotecas como pandas para manipulación de datos, numpy para cálculos numéricos, y plotly/matplotlib para gráficos opcionales. Los scripts son interactivos para facilitar la reproducibilidad. Los datos provienen de la API de SIAR para 11 estaciones baleares (IB01 a IB11; nota: el texto original menciona 12, pero confirmamos 11 activas). Usaré IB05 como ejemplo ilustrativo.

**Requisitos previos**:
- Instala Python y las bibliotecas: `pip install pandas numpy matplotlib plotly`.
- Crea una carpeta `datos_siar_baleares` para almacenar CSVs.
- Ejecuta los scripts en orden secuencial para un flujo completo.
- Para ejecución: Abre una terminal en VS Code (Ctrl+`) y corre `python nombre_script.py`.

---

## 1.2.1 Recopilación Inicial de Datos de Estaciones

El primer paso es obtener metadatos y datos históricos diarios de las estaciones baleares usando la API de SIAR. Esto se hace en dos subpasos: filtrar estaciones y descargar datos diarios.

### Paso 1: Obtención de Metadatos de Estaciones (siarIDIB.py)
- **Descripción**: Este script consulta la API de SIAR para descargar un listado completo de estaciones de España y filtra solo las de Baleares (provincia 'Illes Balears'). Genera un CSV con metadatos útiles para cálculos posteriores (ej. latitud para Ra).
- **Inputs**: Ninguno (usa consulta API interna).
- **Outputs**: `estaciones_baleares.csv` en `datos_siar_baleares/`, con columnas como 'Estacion', 'Codigo', 'Latitud' (en DMS), 'Longitud', 'Altitud', 'Fecha de alta', etc. Resultado: 11 estaciones (IB01-IB11).
- **Ejecución**: 
  - Corre `python siarIDIB.py`.
  - Ejemplo de código simplificado (para referencia, no ejecutar directamente):
    ```python
    import pandas as pd
    # Consulta API (código real en script)
    df_all = pd.read_csv('estaciones_espana.csv')  # Listado completo de SIAR
    baleares_df = df_all[df_all['Provincia'].str.contains('Balears', case=False)]
    baleares_df.to_csv('datos_siar_baleares/estaciones_baleares.csv', index=False)
    print("Metadatos guardados. Estaciones encontradas:", len(baleares_df))
    ```
- **Consejo**: Si la API falla (límite de consultas), usa un backup local de 'estaciones_espana.csv'. Verifica que 'Latitud' esté en formato DMS (grados:minutos:segundos) para conversión posterior.

### Paso 2: Descarga de Datos Diarios por Estación (siarIBconsulta2_corregido.py)
- **Descripción**: Descarga datos históricos diarios (desde ~2004 hasta 2024) para cada estación, manejando límites de la API (descargas mensuales para evitar saturación).
- **Inputs** (interactivo):
  - Código de estación (ej. 'IB05').
  - Fecha inicio (extraída de metadatos o manual: 'dd/mm/aaaa').
  - Fecha fin (por defecto '31/12/2024').
- **Outputs**: `IBXX_datos_completos.csv` en `datos_siar_baleares/`, con columnas como 'Fecha' (YYYY-MM-DD), 'TempMedia' (°C), 'TempMax' (°C), 'HumedadMedia' (%), 'VelViento' (m/s), 'Radiacion' (MJ/m²), 'EtPMon' (mm, ET₀ de SIAR), etc.
- **Ejecución**:
  - Corre `python siarIBconsulta2_corregido.py`.
  - Ingresa inputs cuando se solicite. Ejemplo: Para IB05, descarga incremental si el archivo existe (actualiza solo fechas nuevas).
  - Diagnóstico: Si excede límites API, ejecuta en días diferentes; logs en consola muestran progreso.
- **Resultado para IB05**: ~7000 filas de datos diarios hasta 2024.

**Consejo general para recopilación**: Ejecuta para todas las estaciones en batch (modifica el script a un loop sobre códigos de `estaciones_baleares.csv`). Verifica integridad abriendo CSVs en Excel o pandas.

---

## 1.2.2 Depuración de Datos

Los datos brutos pueden contener NaNs, outliers o errores de medición. Usa `filtra.py` para limpiarlos, asegurando calidad para cálculos posteriores.

### filtra.py (Implementación)
- **Descripción**: Procesa datos brutos para eliminar NaNs y outliers (±3σ de la media por variable). Genera archivos depurados y eliminados, más gráficos para inspección visual.
- **Inputs** (interactivo):
  - Código de estación (ej. 'IB05').
- **Outputs**:
  - `IBXX_datos_depurados.csv`: Datos limpios.
  - `IBXX_datos_eliminados.csv`: NaNs y outliers.
  - Gráficos PNG en `graficas_eliminados/` (ej. `TempMedia_eliminados.png`: scatter con línea de media).
- **Pasos internos**:
  1. Elimina filas con NaNs en columnas de interés: ['TempMedia', 'TempMax', 'TempMin', 'HumedadMedia', 'HumedadMax', 'HumedadMin', 'VelViento', 'DirViento', 'VelVientoMax', 'DirVientoVelMax', 'Radiacion', 'Precipitacion', 'EtPMon'].
  2. Calcula media y σ por variable; filtra filas donde cualquier valor > ±3σ.
- **Ejecución**:
  - Corre `python filtra.py`.
  - Ingresa código de estación. Ejemplo para IB05: Input bruto ~7000 filas → Depurado ~6500 filas (~7% eliminados).
- **Consejo**: Inspecciona gráficos para validar outliers (ej. temperaturas >50°C). Si σ es demasiado estricto, ajusta a ±4σ en el código.

---

## 1.2.3 Cálculo de ET₀ con Penman-Monteith

Verifica ET₀ de SIAR implementando la fórmula FAO-56 Penman-Monteith (PM) en `PM2.py`, incluyendo una variante para cielo claro.

**Fórmula Estándar** (FAO-56):

\[
ET_0 = \frac{0.408 \, \Delta (R_n - G) + \gamma \, \frac{900}{T+273} \, u_2 (e_s - e_a)}{\Delta + \gamma (1+0.34 \, u_2)}
\]

Donde:
- \(R_n\): Radiación neta [MJ/m²/día].
- \(G\): Flujo de calor en suelo [MJ/m²/día] (aprox. 0 para diarios).
- \(T\): TempMedia [°C].
- \(u_2\): VelViento a 2m [m/s].
- \(e_s, e_a\): Presiones de vapor [kPa].
- \(\Delta, \gamma\): Pendiente y constante psicrométrica [kPa/°C].

### PM2.py (Implementación Actualizada)
- **Descripción**: Procesa datos depurados para calcular ET₀ PM estándar (usando Rs medida) y variante cielo claro (Rs = Rso).
- **Inputs** (interactivo):
  - Código de estación (ej. 'IB05').
- **Outputs**: `IBXX_et0_calc.csv` con columnas nuevas: 'J' (día juliano), 'N' (horas sol), 'ET0_calc' (PM estándar), 'diff' (vs EtPMon), 'ET0_sun' (cielo claro), 'diff_sun', 'Ra' (extraterrestre).
- **Pasos internos**:
  - Carga metadatos de `estaciones_baleares.csv` (convierte latitud DMS a decimal).
  - Calcula Ra (usando latitud, día juliano), Rso, Rn, etc.
  - Maneja errores: try/except, drop NaNs; logs en `pm2_log.txt`.
- **Ejecución**:
  - Corre `python PM2.py`.
  - Ingresa código. Ejemplo para IB05: ~6500 filas procesadas, mean diff = 0.12 mm/día.
- **Diagnóstico**: Si mean diff es NaN, verifica NaNs en 'VelViento' o 'Radiacion' (comunes en datos antiguos). Usa prints para debug.

---

## 1.2.4 Estimaciones Alternativas de ET₀

Agrega métodos simplificados (Hargreaves y Valiantzas) usando `variants_et0.py` sobre los outputs de PM2.

### variants_et0.py (Implementación)
- **Descripción**: Calcula ET₀ usando Hargreaves-Samani y Valiantzas (con corrección por humedad).
- **Inputs** (interactivo):
  - Código de estación (ej. 'IB05').
- **Outputs**: `IBXX_et0_variants.csv` con columnas nuevas: 'ET0_harg', 'diff_harg', 'ET0_val', 'diff_val'.
- **Fórmulas** (corregidas para unidades, Ra en MJ/m²/día):
  - Hargreaves: \( ET0_{harg} = 0.0023 \times 0.408 \times Ra \times (T + 17.8) \times (Tmax - Tmin)^{0.5} \)
  - Valiantzas: \( ET0_{val} = 0.408 \times 0.0135 \times 0.338 \times Ra \times (Tmax - Tmin)^{0.3} \times (1 - RH/100)^{0.2} \times (T + 17.8) \) (RH = HumedadMedia).
- **Ejecución**:
  - Corre `python variants_et0.py`.
  - Ingresa código. Ejemplo para IB05: mean diff_harg = -0.45 mm/día (sobreestima en verano); mean diff_val = -0.18 mm/día (mejor en costas húmedas).
- **Consejo**: Ajusta coeficientes (0.0135, 0.338) si calibras para Baleares. Usa HumedadMin si prefieres.

---

## 1.2.5 Cálculo de Errores Comparativos

Evalúa la precisión de los modelos comparando con ET₀ de SIAR usando `calculate_errors.py`.

### calculate_errors_et0.py (Implementación)
- **Descripción**: Calcula errores (MSE, RMSE relativo, MAE, R², AARE) por estación y general (media agregada).
- **Inputs**: Ninguno (procesa todos los `_et0_variants.csv` automáticamente).
- **Outputs**: `analisis_errores.md` con tablas Markdown por estación y general.
- **Fórmulas** (detalladas en script):
  - MSE: \( \frac{1}{n} \sum (x - x_J)^2 \)
  - RMSE: \( \sqrt{MSE} / \bar{x} \)
  - MAE: \( \frac{1}{n} \sum |x - x_J| \)
  - R²: \( \left( \frac{\sum (x - \bar{x})(x_J - \bar{x_J})}{\sqrt{\sum (x - \bar{x})^2 \sum (x_J - \bar{x_J})^2}} \right)^2 \)
  - AARE: \( \frac{1}{n} \sum \left| \frac{x - x_J}{x} \right| \)
- **Ejecución**:
  - Corre `python calculate_errors_et0.py`.
  - Revisa `analisis_errores.md` para tablas. Ejemplo para IB05: RMSE Hargreaves ~0.8 (sobreestima en húmedo); Valiantzas ~0.5 (mejor ajuste).
- **Consejo**: Integra tablas en tu informe. Si datos insuficientes, verifica NaNs en CSVs.

---

