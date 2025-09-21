# 1.2 Depuración y verificación de datos meteorológicos

**Objetivo**

Depurar los datos diarios de las estaciones meteorológicas de las Islas Baleares y, posteriormente, verificar los valores de **ET₀** proporcionados por el SIAR mediante diferentes modelos de cálculo.

---

## 1.2.1 Depuración de datos

La calidad de los datos es fundamental para obtener resultados confiables.  
Por ello, aplicaremos las siguientes transformaciones:

1. **Eliminación de valores nulos (NaN).**  
   - Se eliminan todas las filas que contengan datos faltantes en las columnas de interés.

2. **Detección y filtrado de valores extremos.**  
   - Para cada variable meteorológica relevante, se calcula la **media** y la **desviación estándar (σ)**.  
   - Se eliminan todas las filas que superen **± 3σ** respecto a la media.  
   - Este procedimiento ayuda a reducir el impacto de datos anómalos o errores de medición.

---

## 1.2.2 Cálculo de ET₀ con Penman-Monteith

Una vez depurados los datos, el siguiente paso es verificar los valores de ET₀ proporcionados por el SIAR.  
Para ello, se implementará la fórmula de **FAO Penman-Monteith**, considerada el estándar:

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

---

## 1.2.3 Estimaciones alternativas de ET₀

Además del modelo de Penman-Monteith, se evaluarán métodos simplificados y empíricos.

### 🔹 1.2.3.1 Ecuación original de Hargreaves (1985)

Basada únicamente en radiación solar y temperatura media:

$$
ET_0 = 0.0023 \, R_a \, (T + 17.8) \, (T_{max} - T_{min})^{0.5} \quad (7)
$$

Donde \(R_a\) es la radiación extraterrestre.

---

### 🔹 1.2.3.2 Ecuación de Hargreaves y Samani (1982, 1985)

Incluye la relación entre radiación extraterrestre y diferencia térmica:

$$
ET_0 = 0.0023 \, R_a \, (T + 17.8) \, (T_{max} - T_{min})^{0.5} \quad (8)
$$

El cálculo de \(R_a\) se obtiene de Allen et al. (1998):

$$
R_a = \frac{24 \times 60}{\pi} G_{sc} \left[ d_r \, \big( \omega_s \sin(\phi)\sin(\delta) + \cos(\phi)\cos(\delta)\sin(\omega_s) \big) \right] \quad (10)
$$

Donde:  
- \(G_{sc} = 0.082 \, MJ \, m^{-2} \, min^{-1}\) (constante solar).  
- \(d_r\) = distancia relativa Tierra-Sol:  

$$
d_r = 1 + 0.033 \cos\left(\frac{2 \pi J}{365}\right) \quad (11)
$$

- \(\delta\) = declinación solar:  

$$
\delta = 0.409 \sin\left(\frac{2 \pi J}{365} - 1.39\right) \quad (12)
$$

- \(\omega_s\) = ángulo horario al ocaso:  

$$
\omega_s = \arccos(-\tan(\phi)\tan(\delta)) \quad (13)
$$

Donde:  
- \(J\) = día juliano.  
- \(\phi\) = latitud en radianes.  

---

### 🔹 1.2.3.3 Ecuación de Valiantzas (2017)

Este modelo estima la radiación solar en función de la humedad relativa mínima:

$$
R_s = R_a \, \left[a + b \frac{RH_{min}}{100}\right] \quad (15)
$$

Donde \(a\) y \(b\) son coeficientes empíricos.  
Al combinar esta relación con la ecuación de Hargreaves, se obtiene:

$$
ET_0 = 0.0023 \, R_a \, (T + 17.8) \, (T_{max} - T_{min})^{0.5} \quad (16)
$$

---

## 1.2.4 Comparación de resultados

Se realizará una **comparación sistemática** entre:  
- ET₀ de SIAR.  
- ET₀ calculado con Penman-Monteith.  
- ET₀ estimado con Hargreaves, Hargreaves-Samani y Valiantzas.  

Esto permitirá validar la consistencia de los datos y analizar la aplicabilidad de métodos alternativos en las Islas Baleares.

---
## 21/09/2025: Prrimeras verisones de filtrado

## filtra.py

Recoge IBXX_datos_completos.csv y lo filtra analizando las columnas de variables.

Variables analizadas: El análisis de ±3σ se ejecuta en las columnas de columnas_interes = ['TempMedia', 'TempMax', 'TempMin', 'HumedadMedia', 'HumedadMax', 'HumedadMin', 'VelViento', 'DirViento', 'VelVientoMax', 'DirVientoVelMax', 'Radiacion', 'Precipitacion', 'EtPMon']. Estas son las variables numéricas relevantes para detectar outliers.

**Como funciona:**

- Pregunta por la estación: El script pregunta por el ID de la estación y genera las rutas de archivos basadas en él (ej. IB01_datos_completos.csv).

- Archivos nuevos: Crea IB01_datos_depurados.csv (datos filtrados) y IB01_datos_eliminados.csv (nulos + outliers).

- Gráficas: Para cada variable en columnas_interes, si hay valores eliminados, crea una gráfica scatter de los valores eliminados vs su índice, con una línea horizontal para la media (del original). Las guarda como PNG en graficas_eliminados (ej. IB01_TempMedia_eliminados.png).