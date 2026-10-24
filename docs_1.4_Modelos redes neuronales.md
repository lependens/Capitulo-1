# 1.3 Análisis de Errores y Estimación de ET₀ mediante Redes Neuronales Artificiales (ANN)

> **Lectura vigente — 24/10/2026.** Los apartados anteriores a la última actualización conservan snapshots históricos, incluidas tareas después superadas. La consolidación final y [PROJECT_STATUS](docs/PROJECT_STATUS.md) distinguen cuatro baselines oficiales, preservación pendiente y gates operativos actuales; no se reescriben resultados históricos.

> **Estado vigente — 23/10/2026.** Las secciones y actualizaciones anteriores son snapshots históricos; sus planes, inventarios y afirmaciones de estado pueden estar superados. La actualización final de este documento recoge el estado contrastado, y [PROJECT_STATUS](docs/PROJECT_STATUS.md) es la autoridad de estado rápido. La metodología histórica se conserva sin convertir sus resultados en resultados v1.


Este documento combina el análisis de errores de los modelos empíricos (Penman-Monteith estándar, cielo claro, Hargreaves, Valiantzas, ajustados) con la implementación de **Redes Neuronales Artificiales (ANN)** para estimar la evapotranspiración de referencia (ET₀). Los errores se calculan frente a los valores de referencia de Penman-Monteith (PM) proporcionados por el SIAR, para las estaciones IB01 a IB05. Las métricas evaluadas son:
- **MSE**: Mean Squared Error (mm²/día²)
- **RRMSE**: Root Mean Squared Error relativo (adimensional)
- **MAE**: Mean Absolute Error (mm/día)
- **R²**: Coeficiente de determinación (adimensional)
- **AARE**: Average Absolute Relative Error (adimensional)

Los modelos incluyen PM Estándar, PM Cielo Claro, Hargreaves, Valiantzas, y sus versiones ajustadas con coeficientes AHC. Se comparan los resultados del proyecto en Python (datos hasta 2024, estaciones IB01-IB05) con los del TFG (datos hasta ~2020, estaciones IB01-IB11).

La sección de ANN replica la metodología del TFG, adaptada a Python/Keras, usando validación cruzada por años y métricas consistentes.

---

## 1.3.1 Errores por Estación

### Estación IB01

| Modelo               | MSE (mm²/día²) | RRMSE | MAE (mm/día) | R²   | AARE |
|----------------------|----------------|-------|--------------|------|------|
| PM Estándar          | 0.00           | 0.02  | 0.02         | 1.00 | 0.02 |
| PM Cielo Claro       | 0.53           | 0.23  | 0.49         | 0.93 | 0.17 |
| Hargreaves           | 0.57           | 0.24  | 0.62         | 0.91 | 0.30 |
| Valiantzas           | 0.22           | 0.15  | 0.35         | 0.94 | 0.16 |
| Hargreaves Ajustado  | 0.34           | 0.19  | 0.45         | 0.91 | 0.19 |
| Valiantzas Ajustado  | 0.23           | 0.15  | 0.37         | 0.94 | 0.14 |

### Estación IB02

| Modelo               | MSE (mm²/día²) | RRMSE | MAE (mm/día) | R²   | AARE |
|----------------------|----------------|-------|--------------|------|------|
| PM Estándar          | 0.00           | 0.02  | 0.02         | 1.00 | 0.01 |
| PM Cielo Claro       | 0.42           | 0.20  | 0.41         | 0.94 | 0.14 |
| Hargreaves           | 0.53           | 0.22  | 0.58         | 0.91 | 0.24 |
| Valiantzas           | 0.23           | 0.15  | 0.35         | 0.94 | 0.13 |
| Hargreaves Ajustado  | 0.31           | 0.17  | 0.41         | 0.91 | 0.17 |
| Valiantzas Ajustado  | 0.22           | 0.14  | 0.35         | 0.94 | 0.13 |

### Estación IB03

| Modelo               | MSE (mm²/día²) | RRMSE | MAE (mm/día) | R²   | AARE |
|----------------------|----------------|-------|--------------|------|------|
| PM Estándar          | 0.00           | 0.02  | 0.02         | 1.00 | 0.01 |
| PM Cielo Claro       | 0.44           | 0.21  | 0.44         | 0.94 | 0.15 |
| Hargreaves           | 0.48           | 0.22  | 0.57         | 0.92 | 0.26 |
| Valiantzas           | 0.20           | 0.14  | 0.33         | 0.94 | 0.14 |
| Hargreaves Ajustado  | 0.28           | 0.17  | 0.41         | 0.92 | 0.18 |
| Valiantzas Ajustado  | 0.19           | 0.14  | 0.33         | 0.94 | 0.13 |

### Estación IB04

| Modelo               | MSE (mm²/día²) | RRMSE | MAE (mm/día) | R²   | AARE |
|----------------------|----------------|-------|--------------|------|------|
| PM Estándar          | 0.01           | 0.03  | 0.02         | 1.00 | 0.02 |
| PM Cielo Claro       | 0.54           | 0.25  | 0.52         | 0.95 | 0.19 |
| Hargreaves           | 0.43           | 0.23  | 0.55         | 0.93 | 0.30 |
| Valiantzas           | 0.21           | 0.16  | 0.35         | 0.95 | 0.19 |
| Hargreaves Ajustado  | 0.31           | 0.19  | 0.43         | 0.93 | 0.19 |
| Valiantzas Ajustado  | 0.22           | 0.16  | 0.36         | 0.95 | 0.15 |

### Estación IB05

| Modelo               | MSE (mm²/día²) | RRMSE | MAE (mm/día) | R²   | AARE |
|----------------------|----------------|-------|--------------|------|------|
| PM Estándar          | 0.00           | 0.02  | 0.02         | 1.00 | 0.02 |
| PM Cielo Claro       | 0.52           | 0.25  | 0.49         | 0.93 | 0.18 |
| Hargreaves           | 0.70           | 0.29  | 0.70         | 0.93 | 0.37 |
| Valiantzas           | 0.31           | 0.20  | 0.43         | 0.94 | 0.22 |
| Hargreaves Ajustado  | 0.31           | 0.19  | 0.43         | 0.93 | 0.19 |
| Valiantzas Ajustado  | 0.21           | 0.16  | 0.34         | 0.94 | 0.14 |

## 1.3.2 Errores Generales (Media de Estaciones IB01-IB05)

| Modelo               | MSE (mm²/día²) | RRMSE | MAE (mm/día) | R²   | AARE |
|----------------------|----------------|-------|--------------|------|------|
| PM Estándar          | 0.00           | 0.02  | 0.02         | 1.00 | 0.02 |
| PM Cielo Claro       | 0.48           | 0.22  | 0.46         | 0.94 | 0.16 |
| Hargreaves           | 0.56           | 0.24  | 0.61         | 0.91 | 0.29 |
| Valiantzas           | 0.24           | 0.16  | 0.37         | 0.94 | 0.16 |
| Hargreaves Ajustado  | 0.31           | 0.18  | 0.43         | 0.91 | 0.18 |
| Valiantzas Ajustado  | 0.21           | 0.15  | 0.35         | 0.94 | 0.14 |

## 1.3.3 Comparación con Resultados del TFG

El TFG evaluó los modelos empíricos Hargreaves (HGR), Valiantzas (HG), y Hargreaves con Rs medida (HGRs) frente a PM, usando datos de 11 estaciones (IB01-IB11) hasta ~2020, con filtrado similar (±3σ). El proyecto en Python usa datos hasta 2024 para IB01-IB05. A continuación, se comparan los resultados generales (media de IB01-IB05) para HGR y HG, sin ajustar y ajustados con AHC. HGRs se omite, ya que no está implementado en Python.

### Modelos Empíricos del TFG (Media de IB01-IB05)

**Sin Ajustar**:

| Modelo | MSE (mm²/día²) | RRMSE | MAE (mm/día) | R²   | AARE |
|--------|----------------|-------|--------------|------|------|
| HGRs   | 0.338          | 0.196 | 0.485        | 0.949 | 0.228 |
| HGR    | 0.693          | 0.280 | 0.672        | 0.889 | 0.326 |
| HG     | 0.381          | 0.204 | 0.461        | 0.924 | 0.211 |

**Ajustados**:

| Modelo | MSE (mm²/día²) | RRMSE | MAE (mm/día) | R²   | AARE |
|--------|----------------|-------|--------------|------|------|
| HGRs   | 0.398          | 0.205 | 0.506        | 0.949 | 0.231 |
| HGR    | 0.378          | 0.202 | 0.468        | 0.889 | 0.222 |
| HG     | 0.250          | 0.166 | 0.373        | 0.924 | 0.165 |

### Comparación de Modelos Empíricos (Python vs. TFG)

| Modelo                | Fuente | MSE (mm²/día²) | RRMSE | MAE (mm/día) | R²   | AARE |
|-----------------------|--------|----------------|-------|--------------|------|------|
| Hargreaves            | TFG    | 0.693          | 0.280 | 0.672        | 0.889 | 0.326 |
| Hargreaves            | Python | 0.56           | 0.24  | 0.61         | 0.91  | 0.29  |
| Valiantzas            | TFG    | 0.381          | 0.204 | 0.461        | 0.924 | 0.211 |
| Valiantzas            | Python | 0.24           | 0.16  | 0.37         | 0.94  | 0.16  |
| Hargreaves Ajustado   | TFG    | 0.378          | 0.202 | 0.468        | 0.889 | 0.222 |
| Hargreaves Ajustado   | Python | 0.31           | 0.18  | 0.43         | 0.91  | 0.18  |
| Valiantzas Ajustado   | TFG    | 0.250          | 0.166 | 0.373        | 0.924 | 0.165 |
| Valiantzas Ajustado   | Python | 0.21           | 0.15  | 0.35         | 0.94  | 0.14  |

### Análisis de la Comparación

- **Hargreaves (Sin Ajustar)**: Python reduce el MSE en ~19% (0.56 vs. 0.693), MAE en ~9% (0.61 vs. 0.672), y AARE en ~11% (0.29 vs. 0.326). R² mejora ligeramente (0.91 vs. 0.889). Las mejoras son coherentes debido a una depuración más estricta en Python (filtra.py, ±3σ, ~6500 filas por estación) y datos más recientes (hasta 2024).
- **Valiantzas (Sin Ajustar)**: Python mejora significativamente: MSE ~37% menor (0.24 vs. 0.381), MAE ~20% menor (0.37 vs. 0.461), AARE ~24% menor (0.16 vs. 0.211). R² aumenta (0.94 vs. 0.924). Esto se explica por una posible optimización del coeficiente (0.338 en variants_et0.py) y datos más limpios.
- **Hargreaves Ajustado**: Python reduce MSE ~18% (0.31 vs. 0.378), MAE ~8% (0.43 vs. 0.468), AARE ~19% (0.18 vs. 0.222). R² mejora (0.91 vs. 0.889). Los AHC en Python (0.7675-0.8732) están bien calibrados, partiendo de una base más precisa.
- **Valiantzas Ajustado**: Python mejora MSE ~16% (0.21 vs. 0.250), MAE ~6% (0.35 vs. 0.373), AARE ~15% (0.14 vs. 0.165). R² aumenta (0.94 vs. 0.924). Es el mejor modelo empírico en ambos casos, con Python optimizando aún más gracias a datos depurados.
- **Consistencia**: En ambos, Valiantzas ajustado es el mejor empírico (MAE 0.35-0.373, R² ~0.924-0.94), y el ajuste AHC reduce errores significativamente. Las mejoras en Python son lógicas debido a:
  - **Diferencias temporales**: Datos hasta 2024 en Python podrían incluir años con menos variabilidad o mejor calidad en SIAR.
  - **Filtrado coherente**: Ambos filtran outliers, pero Python (filtra.py) genera gráficos para validación, reduciendo ruido.
  - **Número de estaciones**: TFG usa 11 estaciones, Python solo 5. Incluir IB06-IB11 en Python podría aumentar errores ligeramente (si esas estaciones tienen más variabilidad), acercándose al TFG.
- **Conclusión**: Los resultados de Python son coherentes con el TFG (mismas tendencias), con mejoras esperadas (~6-37% en métricas) debido a datos actualizados y depuración estricta. Para alinear más, genera datos para IB06-IB11 y compara subperíodos comunes (ej. hasta 2020).


---

## ACTUALIZACIÓN 23/10/2026 — ANN histórica y trabajo científico vigente

Los modelos, entrenamientos y métricas anteriores son **HISTÓRICO / LEGACY** y no representan una ANN v1 validada con el nuevo dataset. No se regeneran modelos ni se atribuyen a la nueva arquitectura resultados históricos.

Quality está cerrado con POST corregido; Dataset Observational v1 construido/validado y primera ejecución oficial PM/HS PASS ([evidencia](scientific_pipeline/results/pm-vpd-20261023T194553Z/README.md)). Antes de nuevos modelos: auditar Valiantzas → VALIANTZAS_V1 → comparar baselines clásicos; después estadística/regresión, árboles, boosting y ANN.

Cualquier aprendizaje futuro exige split cronológico TRAIN < VALIDATION < TEST, transformaciones y parámetros aprendidos solo en TRAIN. No reutilizar AHC ni filtros ajustados sobre toda la serie como evidencia fuera de muestra. [ROADMAP](docs/ROADMAP.md).


---

## ACTUALIZACIÓN 24/10/2026 — Cuatro referencias clásicas antes de nuevos modelos

Los modelos ANN anteriores siguen HISTÓRICO / LEGACY. Dataset v1 y **PM_FAO56_V1 / HS_RA_V1 / VALIANTZAS_V1 / HG_RS_V1**, con comparación común n=70.860, ya están PASS ([resultados](scientific_pipeline/results/four-classical-20261024T042250Z/README.md)). No queda implementar esos baselines como etapa previa.

El siguiente orden es **preservación de artefactos científicos → AHC-METHODOLOGY-REVIEW → contrato de evaluación temporal → modelos calibrados simples → regresión/árboles/boosting → ANN después**. No se inicia entrenamiento ni se reetiquetan modelos históricos.

AHC legacy full-series/in-sample no demuestra generalización. La misión AHC específica definirá usos climatológicos/predictivos y validación antes de AHC_V1; esta actualización no decide todo el dataset frente a TRAIN. ML mantiene separación temporal sin leakage. [ROADMAP](docs/ROADMAP.md).
