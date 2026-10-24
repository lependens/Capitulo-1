> Nota histórica: la aplicación Dash asociada a este análisis se desplegó en Render y fue retirada en octubre de 2026.

> **Lectura vigente — 24/10/2026.** Los apartados anteriores a la última actualización conservan snapshots históricos, incluidas tareas después superadas. La consolidación final y [PROJECT_STATUS](docs/PROJECT_STATUS.md) distinguen cuatro baselines oficiales, preservación pendiente y gates operativos actuales; no se reescriben resultados históricos.

> **Estado vigente — 23/10/2026.** Las secciones y actualizaciones anteriores son snapshots históricos; sus planes, inventarios y afirmaciones de estado pueden estar superados. La actualización final de este documento recoge el estado contrastado, y [PROJECT_STATUS](docs/PROJECT_STATUS.md) es la autoridad de estado rápido. La metodología histórica se conserva sin convertir sus resultados en resultados v1.




# 1.3 Análisis de Errores

Este apartado presenta los resultados de los errores de los modelos de evapotranspiración (ET₀) calculados frente a los valores de referencia de Penman-Monteith (PM) proporcionados por el SIAR, para las estaciones IB01 a IB06. Las métricas evaluadas son:
- **MSE**: Mean Squared Error (mm²/día²)
- **RRMSE**: Root Mean Squared Error relativo (adimensional)
- **MAE**: Mean Absolute Error (mm/día)
- **R²**: Coeficiente de determinación (adimensional)
- **AARE**: Average Absolute Relative Error (adimensional)

Los modelos incluyen PM Estándar, PM Cielo Claro, Hargreaves Ra (con radiación extraterrestre Ra), Hargreaves Rs (con radiación solar medida Rs), Valiantzas, y sus versiones ajustadas con coeficientes AHC. Se comparan los resultados del proyecto en Python (datos hasta 2024, estaciones IB01-IB06) con los del TFG (datos hasta ~2020, estaciones IB01-IB11).

---

## 1.3.1 Errores por Estación

### Estación IB01

| Modelo | MSE (mm²/día²) | RRMSE | MAE (mm/día) | R²   | AARE |
|--------|----------------|-------|--------------|------|------|
| PM Estándar | 0.0 | 0.02 | 0.02 | 1.0 | 0.02 |
| PM Cielo Claro | 0.53 | 0.23 | 0.49 | 0.93 | 0.17 |
| Hargreaves Ra | 0.57 | 0.24 | 0.62 | 0.91 | 0.3 |
| Hargreaves Rs | 0.23 | 0.15 | 0.41 | 0.96 | 0.19 |
| Valiantzas | 0.22 | 0.15 | 0.35 | 0.94 | 0.16 |
| Hargreaves Ra Ajustado | 0.34 | 0.19 | 0.45 | 0.91 | 0.19 |
| Hargreaves Rs Ajustado | 0.11 | 0.11 | 0.27 | 0.96 | 0.14 |
| Valiantzas Ajustado | 0.23 | 0.15 | 0.37 | 0.94 | 0.14 |

### Estación IB02

| Modelo | MSE (mm²/día²) | RRMSE | MAE (mm/día) | R²   | AARE |
|--------|----------------|-------|--------------|------|------|
| PM Estándar | 0.0 | 0.02 | 0.02 | 1.0 | 0.01 |
| PM Cielo Claro | 0.42 | 0.2 | 0.41 | 0.94 | 0.14 |
| Hargreaves Ra | 0.53 | 0.22 | 0.58 | 0.91 | 0.24 |
| Hargreaves Rs | 0.19 | 0.13 | 0.36 | 0.96 | 0.16 |
| Valiantzas | 0.23 | 0.15 | 0.35 | 0.94 | 0.13 |
| Hargreaves Ra Ajustado | 0.31 | 0.17 | 0.41 | 0.91 | 0.17 |
| Hargreaves Rs Ajustado | 0.15 | 0.12 | 0.32 | 0.96 | 0.14 |
| Valiantzas Ajustado | 0.22 | 0.14 | 0.35 | 0.94 | 0.13 |

### Estación IB03

| Modelo | MSE (mm²/día²) | RRMSE | MAE (mm/día) | R²   | AARE |
|--------|----------------|-------|--------------|------|------|
| PM Estándar | 0.0 | 0.02 | 0.02 | 1.0 | 0.01 |
| PM Cielo Claro | 0.44 | 0.21 | 0.44 | 0.94 | 0.15 |
| Hargreaves Ra | 0.48 | 0.22 | 0.57 | 0.92 | 0.26 |
| Hargreaves Rs | 0.19 | 0.14 | 0.36 | 0.96 | 0.17 |
| Valiantzas | 0.2 | 0.14 | 0.33 | 0.94 | 0.14 |
| Hargreaves Ra Ajustado | 0.28 | 0.17 | 0.41 | 0.92 | 0.18 |
| Hargreaves Rs Ajustado | 0.15 | 0.12 | 0.32 | 0.96 | 0.15 |
| Valiantzas Ajustado | 0.19 | 0.14 | 0.33 | 0.94 | 0.14 |

### Estación IB04

| Modelo | MSE (mm²/día²) | RRMSE | MAE (mm/día) | R²   | AARE |
|--------|----------------|-------|--------------|------|------|
| PM Estándar | 0.0 | 0.02 | 0.02 | 1.0 | 0.01 |
| PM Cielo Claro | 0.44 | 0.22 | 0.44 | 0.94 | 0.16 |
| Hargreaves Ra | 0.63 | 0.26 | 0.65 | 0.91 | 0.3 |
| Hargreaves Rs | 0.25 | 0.17 | 0.41 | 0.96 | 0.2 |
| Valiantzas | 0.24 | 0.16 | 0.36 | 0.95 | 0.15 |
| Hargreaves Ra Ajustado | 0.33 | 0.19 | 0.44 | 0.91 | 0.19 |
| Hargreaves Rs Ajustado | 0.16 | 0.13 | 0.33 | 0.96 | 0.15 |
| Valiantzas Ajustado | 0.22 | 0.15 | 0.34 | 0.95 | 0.13 |

### Estación IB05

| Modelo | MSE (mm²/día²) | RRMSE | MAE (mm/día) | R²   | AARE |
|--------|----------------|-------|--------------|------|------|
| PM Estándar | 0.0 | 0.02 | 0.02 | 1.0 | 0.02 |
| PM Cielo Claro | 0.52 | 0.25 | 0.49 | 0.93 | 0.18 |
| Hargreaves Ra | 0.7 | 0.29 | 0.7 | 0.93 | 0.37 |
| Hargreaves Rs | 0.35 | 0.21 | 0.51 | 0.96 | 0.26 |
| Valiantzas | 0.31 | 0.2 | 0.43 | 0.94 | 0.22 |
| Hargreaves Ra Ajustado | 0.31 | 0.19 | 0.43 | 0.93 | 0.19 |
| Hargreaves Rs Ajustado | 0.12 | 0.12 | 0.27 | 0.96 | 0.16 |
| Valiantzas Ajustado | 0.21 | 0.16 | 0.34 | 0.94 | 0.14 |

### Estación IB06

| Modelo | MSE (mm²/día²) | RRMSE | MAE (mm/día) | R²   | AARE |
|--------|----------------|-------|--------------|------|------|
| PM Estándar | 0.01 | 0.03 | 0.02 | 1.0 | 0.02 |
| PM Cielo Claro | 0.48 | 0.23 | 0.46 | 0.93 | 0.17 |
| Hargreaves Ra | 0.98 | 0.33 | 0.84 | 0.92 | 0.4 |
| Hargreaves Rs | 0.26 | 0.17 | 0.43 | 0.96 | 0.21 |
| Valiantzas | 0.28 | 0.18 | 0.4 | 0.94 | 0.2 |
| Hargreaves Ra Ajustado | 0.29 | 0.18 | 0.42 | 0.92 | 0.2 |
| Hargreaves Rs Ajustado | 0.13 | 0.12 | 0.3 | 0.96 | 0.17 |
| Valiantzas Ajustado | 0.2 | 0.15 | 0.34 | 0.94 | 0.15 |

## 1.3.2 Errores Generales

| Modelo | MSE (mm²/día²) | RRMSE | MAE (mm/día) | R²   | AARE |
|--------|----------------|-------|--------------|------|------|
| PM Estándar | 0.0 | 0.02 | 0.02 | 1.0 | 0.02 |
| PM Cielo Claro | 0.48 | 0.23 | 0.46 | 0.93 | 0.17 |
| Hargreaves Ra | 0.62 | 0.26 | 0.65 | 0.91 | 0.31 |
| Hargreaves Rs | 0.24 | 0.16 | 0.41 | 0.96 | 0.2 |
| Valiantzas | 0.25 | 0.16 | 0.37 | 0.94 | 0.17 |
| Hargreaves Ra Ajustado | 0.31 | 0.18 | 0.42 | 0.91 | 0.18 |
| Hargreaves Rs Ajustado | 0.13 | 0.12 | 0.29 | 0.96 | 0.15 |
| Valiantzas Ajustado | 0.21 | 0.15 | 0.35 | 0.94 | 0.14 |

## 1.3.3 Coeficientes de Ajuste (AHC) por Estación

| Estacion   |   AHC_Hargreaves_Ra |   AHC_Hargreaves_Rs |   AHC_Valiantzas |
|:-----------|--------------------:|--------------------:|-----------------:|
| IB01       |              0.8288 |              0.9164 |           0.9162 |
| IB02       |              0.8732 |              0.9742 |           0.9785 |
| IB03       |              0.8578 |              0.9634 |           0.9674 |
| IB04       |              0.8124 |              0.9094 |           0.877  |
| IB05       |              0.7675 |              0.8506 |           0.8499 |
| IB06       |              0.757  |              0.9237 |           0.8734 |
| General    |              0.8161 |              0.923  |           0.9104 |

## 1.3.4 Comparación con TFG

### Medias de Errores Comparadas

| Modelo                | Fuente | MSE (mm²/día²) | RRMSE | MAE (mm/día) | R²    | AARE  |
|-----------------------|--------|----------------|-------|--------------|-------|-------|
| Hargreaves Ra         | TFG    | 0.693          | 0.268 | 0.672        | 0.889 | 0.326 |
| Hargreaves Ra         | Python | 0.62           | 0.26  | 0.65         | 0.91  | 0.31  |
| Hargreaves Rs         | TFG    | N/A            | N/A   | N/A          | N/A   | N/A   |
| Hargreaves Rs         | Python | 0.24           | 0.16  | 0.41         | 0.96  | 0.2   |
| Valiantzas            | TFG    | 0.381          | 0.202 | 0.461        | 0.924 | 0.211 |
| Valiantzas            | Python | 0.25           | 0.16  | 0.37         | 0.94  | 0.17  |
| Hargreaves Ra Ajustado| TFG    | 0.378          | 0.202 | 0.468        | 0.889 | 0.222 |
| Hargreaves Ra Ajustado| Python | 0.31           | 0.18  | 0.42         | 0.91  | 0.18  |
| Hargreaves Rs Ajustado| TFG    | N/A            | N/A   | N/A          | N/A   | N/A   |
| Hargreaves Rs Ajustado| Python | 0.13           | 0.12  | 0.29         | 0.96  | 0.15  |
| Valiantzas Ajustado   | TFG    | 0.250          | 0.166 | 0.373        | 0.924 | 0.165 |
| Valiantzas Ajustado   | Python | 0.21           | 0.15  | 0.35         | 0.94  | 0.14  |

### Análisis de la Comparación

- **Hargreaves Ra (Sin Ajustar)**: Python reduce el MSE en ~10% (0.62 vs. 0.693), MAE en ~3% (0.65 vs. 0.672), y AARE en ~5% (0.31 vs. 0.326). R² mejora ligeramente (0.91 vs. 0.889). Las mejoras son coherentes debido a una depuración más estricta en Python (filtra.py, ±3σ, ~6500 filas por estación) y datos más recientes (hasta 2024).
- **Hargreaves Rs (Sin Ajustar)**: No disponible en TFG (HGRs mencionado pero no implementado). En Python, muestra mejor rendimiento (MSE 0.24, MAE 0.41), destacando la ventaja de usar Rs medida.
- **Valiantzas (Sin Ajustar)**: Python mejora significativamente: MSE ~34% menor (0.25 vs. 0.381), MAE ~20% menor (0.37 vs. 0.461), AARE ~19% menor (0.17 vs. 0.211). R² aumenta (0.94 vs. 0.924). Esto se explica por una posible optimización del coeficiente (0.338 en variants_et0.py) y datos más limpios.
- **Hargreaves Ra Ajustado**: Python reduce MSE ~18% (0.31 vs. 0.378), MAE ~10% (0.42 vs. 0.468), AARE ~19% (0.18 vs. 0.222). R² mejora (0.91 vs. 0.889). Los AHC en Python (0.7570-0.8732) están bien calibrados, partiendo de una base más precisa.
- **Hargreaves Rs Ajustado**: No disponible en TFG. En Python, es el mejor modelo empírico (MSE 0.13, MAE 0.29, R² 0.96), con AHC (0.8506-0.9742) reduciendo errores drásticamente.
- **Valiantzas Ajustado**: Python mejora MSE ~16% (0.21 vs. 0.250), MAE ~6% (0.35 vs. 0.373), AARE ~15% (0.14 vs. 0.165). R² aumenta (0.94 vs. 0.924). Es uno de los mejores modelos empíricos en ambos casos, con Python optimizando aún más gracias a datos depurados.
- **Consistencia**: En ambos, Valiantzas ajustado es fuerte, pero en Python, Hargreaves Rs ajustado supera (MAE 0.29 vs. 0.35 para Valiantzas ajustado), validando la inclusión de Rs. El ajuste AHC reduce errores significativamente. Las mejoras en Python son lógicas debido a:
  - **Diferencias temporales**: Datos hasta 2024 en Python podrían incluir años con menos variabilidad o mejor calidad en SIAR.
  - **Filtrado coherente**: Ambos filtran outliers, pero Python (filtra.py) genera gráficos para validación, reduciendo ruido.
  - **Número de estaciones**: TFG usa 11 estaciones, Python ahora 6. Incluir IB07-IB11 en Python podría aumentar errores ligeramente (si esas estaciones tienen más variabilidad), acercándose al TFG.
- **Conclusión**: Los resultados de Python son coherentes con el TFG (mismas tendencias), con mejoras esperadas (~3-34% en métricas) debido a datos actualizados, depuración estricta y nuevo modelo Hargreaves Rs. Para alinear más, genera datos para IB07-IB11 y compara subperíodos comunes (ej. hasta 2020).

## Análisis por Modelo

- **Hargreaves Ra**:
  - **Python (MAE)**: 0.48-0.98, promedio 0.65. Sobreestima en estaciones secas (IB06 MAE 0.84).
  - **TFG (MAE)**: 0.672. Python similar, con leve mejora.
  - **R²**: Python (0.91-0.93) vs. TFG (0.889), estable.

- **Hargreaves Rs**:
  - **Python (MAE)**: 0.19-0.35, promedio 0.41. Mejor que Ra, gracias a Rs medida (precisa en estaciones con datos radiométricos).
  - **TFG**: No implementado (HGRs pendiente).
  - **R²**: Python (0.96), alto, indica buena correlación.

- **Valiantzas**:
  - **Python (MAE)**: 0.20-0.31, promedio 0.37. Sensibilidad a humedad (RH) podría beneficiarse de datos más recientes (2020-2024).
  - **TFG (MAE)**: 0.461. Python mejora ~20%.
  - **R²**: Python (0.94) vs. TFG (0.924), mejora consistente.

- **Hargreaves Ra Ajustado**:
  - **Python (MAE)**: 0.28-0.33, promedio 0.42. AHC (0.7570-0.8732) reduce errores.
  - **TFG (MAE)**: 0.468. Python mejora ~10%.
  - **R²**: Python (0.91-0.93) vs. TFG (0.889), mejora leve.

- **Hargreaves Rs Ajustado**:
  - **Python (MAE)**: 0.12-0.16, promedio 0.29. Mejor modelo empírico general.
  - **TFG**: No disponible.
  - **R²**: Python (0.96), excelente.

- **Valiantzas Ajustado**:
  - **Python (MAE)**: 0.19-0.23, promedio 0.35.
  - **TFG (MAE)**: 0.373. Python mejora ~6%.
  - **R²**: Python (0.94) vs. TFG (0.924), consistente.

### Análisis General

- **Consistencia**: En ambos, Valiantzas ajustado es fuerte, pero Hargreaves Rs ajustado destaca en Python (MAE 0.29, R² 0.96), seguido de Valiantzas ajustado (MAE 0.35). El ajuste AHC reduce errores (~20-50%), validando la implementación en Python (`calculate_ahc.py`).
- **Mejoras en Python**: Reducción de errores (~3-34% en MAE, MSE, AARE) debido a:
  - **Datos recientes**: Python usa datos hasta 2024 (vs. ~2020 en TFG), con posible mejor calidad en SIAR (menos NaNs).
  - **Filtrado estricto**: Ambos filtran outliers (±3σ), pero Python (`filtra.py`) genera gráficos para validación, reduciendo ruido (~6500 filas por estación).
  - **Estaciones**: TFG promedia 11 estaciones, Python ahora 6. IB07-IB11 (posiblemente interiores) podrían aumentar errores en TFG (ej. mayor variabilidad en RH para Valiantzas).
- **Diferencias temporales**: Los 5 años adicionales en Python podrían capturar cambios climáticos (ej. veranos más cálidos), mejorando AHC (0.7570-0.8732 para Hargreaves Ra, 0.8506-0.9742 para Rs, 0.8499-0.9785 para Valiantzas).
- **Limitaciones**: La ausencia de IB07-IB11 en Python podría sesgar la media hacia estaciones "fáciles" (ej. costeras). HGRs (TFG) ahora implementado como Hargreaves Rs en Python.

### Conclusiones y Recomendaciones

- Los resultados de Python son coherentes con el TFG, con mejoras esperadas debido a datos más recientes, filtrado robusto y el nuevo modelo Hargreaves Rs (mejor empírico: MAE 0.29 ajustado).
- Hargreaves Rs ajustado destaca como el mejor empírico en Python, superando a Valiantzas ajustado (MAE 0.29 vs. 0.35), con Python optimizando aún más (vs. TFG MAE 0.373 para Valiantzas).
- Para alinear con el TFG:
  - Procesar IB07-IB11 con `siarIBconsulta2_corregido.py`, `filtra.py`, `PM2.py`, `variants_et0.py`, y `calculate_ahc.py`.
  - Comparar subperíodos comunes (ej. hasta 2020) para aislar el efecto temporal.
  - Verificar HGRs del TFG vs. Hargreaves Rs en Python para completar la comparación.
- Futuros análisis podrían explorar AHC estacionales o por región para mejorar la precisión.
---

## ACTUALIZACIÓN 23/10/2026 — Alcance de este análisis histórico

Las tablas, AHC y comparaciones anteriores permanecen **HISTÓRICO / LEGACY**. No se reetiquetan como resultados v1 ni se usan para inferir mejoras causales. Pearson² y el RRMSE histórico no deben confundirse con R² predictivo y RMSE en mm/día; AHC dentro de muestra no demuestra generalización.

Quality POST corregido y Dataset Observational v1 ya están construidos/validados. Posteriormente la misión **PM-VPD-FLOAT-TOLERANCE-AND-BASELINES = PASS** publicó mediante PR #31 [resultados oficiales PM_FAO56_V1 / HS_RA_V1](scientific_pipeline/results/pm-vpd-20261023T194553Z/README.md). Ese informe, sus métricas y manifiesto son la autoridad actual; esta actualización solo enlaza la evidencia, sin añadir tablas ni reinterpretar los resultados anteriores.

Valiantzas histórico sigue LEGACY / FUTURE BASELINE REVIEW. No es un tercer resultado v1. La futura comparación exige auditoría metodológica, implementación reproducible y cohortes explícitas. Las [limitaciones vigentes](docs/PROJECT_STATUS.md) incluyen EtPMon calculada SIAR, IB09 excluida por Quality FAIL y procedencia instrumental parcial.


## ACTUALIZACIÓN 24/10/2026 — Tres baselines clásicos oficiales

**VALIANTZAS-V1-AND-CLASSICAL-BASELINES-EXPORT = PASS**. Supera el estado
Valiantzas LEGACY / FUTURE BASELINE REVIEW de la actualización del 23/10;
el texto y resultados históricos anteriores se conservan como tales.

Se adopta Martí et al. (2024), Ecs.1b+3, con factor 0.408 y coeficiente
expandido **0.001861704**, Ra en MJ/m²/día, `(1.001−RHmedia/100)^0.2` y
Tcalc de extremos. [Decisión, fuentes y unidades](scientific_pipeline/docs/valiantzas_v1.md).
No es transcripción literal de Ec.4 ni copia del factor 1−RH/100 legacy.
Sin calibración local ni sustitución silenciosa de TempMedia original.

Dataset SHA256 `0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb`, intacto. Pipeline `4ac52e25c803cb2301316c5cb1353bf30cdb427a`
([PR #33](https://github.com/lependens/Capitulo-1/pull/33)); 68 tests y CI Python 3.11/3.12/3.13 PASS.
PM/HS mantienen exactamente predicciones y elegibilidad oficiales anteriores.
VAL: 70.944 predicciones, 70.878 pares propios, 0 fallos.

[Informe de ejecución, cohortes, desgloses, CSV y hashes](scientific_pipeline/results/classical-20261024T034529Z/README.md).
EtPMon sigue siendo referencia calculada SIAR, no validación instrumental
independiente. Se retienen IB09 Quality FAIL, IB101 corto, unidades/procedencia
parcial y convención de viento 2 m; raw/Quality/Dataset sin cambios.


Resultados reales v1, separados de las tablas legacy:

| Grupo | Cohorte | Método | n | MAE | RMSE | bias/MBE | R² predictivo |
|---|---|---|---:|---:|---:|---:|---:|
| all | own | PM | 70861 | 0.011540 | 0.028298 | -0.001565 | 0.999735 |
| all | own | HS-Ra | 70912 | 0.712895 | 0.874488 | 0.461359 | 0.746479 |
| all | own | VAL | 70878 | 0.455990 | 0.609080 | 0.168851 | 0.877013 |
| all | PM_HS | PM | 70861 | 0.011540 | 0.028298 | -0.001565 | 0.999735 |
| all | PM_HS | HS-Ra | 70861 | 0.712850 | 0.874365 | 0.461603 | 0.746552 |
| all | PM_VAL | PM | 70860 | 0.011540 | 0.028298 | -0.001565 | 0.999735 |
| all | PM_VAL | VAL | 70860 | 0.455952 | 0.608984 | 0.168919 | 0.877055 |
| all | HS_VAL | HS-Ra | 70878 | 0.712850 | 0.874422 | 0.461455 | 0.746515 |
| all | HS_VAL | VAL | 70878 | 0.455990 | 0.609080 | 0.168851 | 0.877013 |
| all | PM_HS_VAL | PM | 70860 | 0.011540 | 0.028298 | -0.001565 | 0.999735 |
| all | PM_HS_VAL | HS-Ra | 70860 | 0.712843 | 0.874359 | 0.461627 | 0.746559 |
| all | PM_HS_VAL | VAL | 70860 | 0.455952 | 0.608984 | 0.168919 | 0.877055 |

La comparación principal usa PM_HS_VAL. No se ajustan parámetros ni se presenta
el acuerdo con SIAR como exactitud frente a ET medida. Estaciones, temporadas,
periodos y extremos se interpretan en el informe enlazado.


## ACTUALIZACIÓN 24/10/2026 — Hargreaves-Rs y cuatro baselines oficiales

**HARGREAVES-RS-V1-AND-FOUR-BASELINES = PASS**.
Se incorpora HG_RS_V1: **0.0135×0.408×Radiacion×(Tcalc+17.8)**, Rs medida en
MJ/m²/día, salida mm/día, Tcalc=(Tmax+Tmin)/2. Sin Ra, √TD, humedad, viento
ni calibración local. Legacy usaba TempMedia; se conservan su código y resultados.
[Contrato y fuentes](scientific_pipeline/docs/hargreaves_rs_v1.md).

HG-Rs: 70978 predicciones, 70912 evaluables, 0 fallos.
Cohorte común de cuatro: **70860**. PM/HS/VAL conservan exactamente todas
sus predicciones, elegibilidad, métricas y hashes de cohortes anteriores.
Pipeline `a0b440fa3ab92e4593be521e4e819c6421327656`, [PR #35](https://github.com/lependens/Capitulo-1/pull/35); 75 tests y CI PASS.
Dataset SHA256 `0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb` sin cambios.

[Resultados completos, cohortes, estaciones, temporadas, CSV y hashes](scientific_pipeline/results/four-classical-20261024T042250Z/README.md).
EtPMon calculada SIAR comparte Rs con PM/HG-Rs; no es validación instrumental
independiente. Se conservan procedencia parcial, viento convencional 2 m,
IB09 FAIL, IB101 corto y límites empíricos. Raw/Quality/Dataset sin cambios.


Resultados reales de cohortes propias y comparación principal de cuatro:

| Grupo | Cohorte | Método | n | MAE | RMSE | bias/MBE | R² predictivo |
|---|---|---|---:|---:|---:|---:|---:|
| all | own | PM | 70861 | 0.011540 | 0.028298 | -0.001565 | 0.999735 |
| all | own | HS-Ra | 70912 | 0.712895 | 0.874488 | 0.461359 | 0.746479 |
| all | own | VAL | 70878 | 0.455990 | 0.609080 | 0.168851 | 0.877013 |
| all | own | HG-Rs | 70912 | 0.471178 | 0.565634 | 0.275611 | 0.893934 |
| all | PM_HS_VAL_HGRS | PM | 70860 | 0.011540 | 0.028298 | -0.001565 | 0.999735 |
| all | PM_HS_VAL_HGRS | HS-Ra | 70860 | 0.712843 | 0.874359 | 0.461627 | 0.746559 |
| all | PM_HS_VAL_HGRS | VAL | 70860 | 0.455952 | 0.608984 | 0.168919 | 0.877055 |
| all | PM_HS_VAL_HGRS | HG-Rs | 70860 | 0.471171 | 0.565637 | 0.275804 | 0.893935 |

Los resultados históricos y de tres modelos anteriores permanecen preservados.
No se interpretan como una comparación de cuatro ni como validación de AHC.


---

## CONSOLIDACIÓN 24/10/2026 — Interpretación de cuatro métodos y límites AHC

Las tablas oficiales de cuatro anteriores se conservan como resultados publicados sobre Dataset v1, distintas de las tablas legacy y del AHC histórico. Autoridad de métricas/desgloses/hashes: [publicación de cuatro métodos](scientific_pipeline/results/four-classical-20261024T042250Z/README.md), PR #36; cohorte común PM_HS_VAL_HGRS **70.860**.

HG-Rs mejora RMSE/R² respecto a VAL; VAL mejora MAE y bias absoluto. No se declara vencedor universal ni validación independiente frente a ET₀ instrumental: EtPMon es referencia calculada SIAR. HS-Ra usa temperatura/geometría solar; VAL añade humedad media; HG-Rs usa Rs medida; PM integra también humedad, viento y presión/altitud. La similitud de inputs puede favorecer concordancia, interpretación sin causalidad demostrada. Deben compararse también input requirements / operational availability y precisión alcanzable con menos variables.

**AHC legacy:** mean(ET0_calc/ET0_empirical) por estación, calculado sobre toda la serie; multiplicación AHC×ET0_empirical y evaluación sobre los mismos días. Es HISTÓRICO / full-series / in-sample, no validación predictiva. Ningún baseline v1 incorpora ese ajuste.

**AHC-METHODOLOGY-REVIEW pendiente:** distinguir coeficiente climatológico de estación de calibración predictiva, revisar bibliografía/estabilidad temporal y escalas anual/estacional/mensual antes de definir validación temporal e implementar AHC_V1. Si se estima de datos es parámetro calibrado; no se decide aquí todo el dataset frente a solo TRAIN ni se cierra la metodología.

Antes de nuevos modelos, **PROJECT-ARTIFACT-BACKUP-V1** debe resolver preservación del dataset y outputs masivos que hoy permanecen fuera de Git. [Ubicaciones y CSV de 75.496 filas](docs/DATA_LOCATIONS.md).

---

## CIERRE CIENTÍFICO AHC — 24/10/2026

**AHC-METHODOLOGY-AND-TEMPORAL-VALIDATION = PASS.** El pendiente metodológico
anterior queda cerrado por [protocolo y bibliografía](scientific_pipeline/docs/ahc_methodology.md)
y [ejecución oficial](scientific_pipeline/results/ahc-20261024T172400Z/README.md).
No se reescriben coeficientes ni métricas históricas del TFG.

AHC v1 es multiplicador OLS sin intercepto por estación contra PM_FAO56_V1:
a=Σ(x·PM)/Σ(x²). Para HS, coeficiente físico 0.0023·a; a es adimensional.
La media de cocientes legacy responde a otro estimando y queda como sensibilidad,
sin escoger fórmula por TEST. PM/EtPMon son referencias calculadas, no ET medida.

TRAIN 2004–2015 ajusta; VALIDATION 2016–2020 selecciona complejidad;
TEST 2021–2025 evalúa sin refit. TEST retrospectivo: los baselines de todo el
histórico ya eran conocidos. No se declara validación prospectiva independiente.
TEST común de siete métodos contra PM, n=15.958: RMSE HS 0.948651→0.638838,
HG-Rs 0.564675→0.414241, VAL 0.628676→0.523955 mm/día. HG-Rs estacional,
elegido solo por VALIDATION, RMSE TEST 0.401392; HS y VAL conservan anual.

Estabilidad robusta por bloques PASS, CV máximo 6,61 %. Esto no garantiza
estabilidad del error: HS en IB11/2025 empeora RMSE 0.756431→0.896925.
Se conservan métricas por estación/periodo, sin ajustar parámetros tras TEST.
AHC_CLIMATOLOGY se obtiene después con el histórico hasta 2025 y nunca sustituye
AHC_VALIDATED en tablas fuera de muestra. IB101: climatología descriptiva,
sin coeficiente validado robusto; IB09: FAIL; IB07: sin VALIDATION/TEST.

Pipeline `38503dbe8422982deeb3680bbc3e82fd950da4e1`, PR #38,
86 tests PASS y CI 3.11/3.12/3.13 PASS. Raw, Quality y Dataset v1 intactos.
Contratos ML/visualización definidos, sin entrenar ML/ANN ni construir Web.
La preservación externa de artefactos masivos sigue pendiente; este cierre
científico no certifica PROJECT-ARTIFACT-BACKUP-V1. [Handoff para 03](scientific_pipeline/docs/ahc_handoff_03.md).
