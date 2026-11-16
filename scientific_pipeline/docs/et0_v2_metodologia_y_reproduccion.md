# ET₀ V2: metodología y reproducción

> Borrador incompleto: la prueba independiente de inferencia y la restauración final están pendientes. No utilizar como certificado de reproducción.

## Diseño

El protocolo [`et0-v2-preparation.json`](../experiments/et0-v2-preparation.json) fija Dataset v1 SHA256 `0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb`, salida objetivo PM_FAO56_V1 calculada, DEVELOPMENT 2004–2019 y retrospectiva 2020–2025. La referencia es un cálculo, no una medida instrumental independiente. IB09 se admite únicamente en DEVELOPMENT por excepción de Dirección; Quality V1 sigue FAIL y la verificación oficial de periodo sigue pendiente.

Las familias comparten las entradas entre métodos dentro de cada comparación: HS-Ra utiliza Tmin, Tmax y Ra; HG-Rs utiliza la media de extremos y Rs; Valiantzas utiliza Tmin, Tmax, humedad media y Ra. `et0_v2._inputs` construye esas entradas y `et0_v2_support.design_vector` incorpora indicadores del calendario estacional o mensual.

AHC utiliza mínimos cuadrados por el origen y exige suma de cuadrados positiva. La selección exterior deja un año de DEVELOPMENT fuera en cada fold; el criterio principal es la media no ponderada del RMSE anual. El coeficiente del histórico completo se informa aparte como in-sample. ML evalúa siete familias de algoritmos y 37 candidatos por celda. ANN A usa el último 15 % del entrenamiento exterior como validación interna; ANN E usa una muestra del 15 % ordenada por hash. Ambas comparan cinco arquitecturas y usan tres semillas; la salida final promedia las tres redes.

## Componentes ejecutados

| Archivo y función | Entrada y operación | Salida y control |
|---|---|---|
| `et0_v2.prepare` | Dataset y handoff IB09; verifica identidad y construye filas DEVELOPMENT | Vista e inventario, sin incluir filas retrospectivas. |
| `et0_v2_support.outer_plan` | Fechas elegibles de una estación y familia | Folds por año de DEVELOPMENT y soporte. |
| `et0_v2_models.ahc_coefficients` | Grupos, estimaciones empíricas y referencia PM | Coeficientes por grupo, con soporte explícito. |
| `et0_v2_models.fit_candidate` | Celda, folds y configuración congelada | Ajustes y puntuaciones de desarrollo; no usa TEST. |
| `et0_v2_campaign.fit_batch` | Cápsula, protocolo y checkpoints | Resultados incrementales por celda y candidato. |
| `et0_v2_campaign.seal_stage` y `freeze` | Fases y artefactos existentes | Sellos verificados y freeze global. |
| `et0_v2_retrospective.evaluate` | Dataset, cápsula, modelos y SHA del freeze | Journal de acceso, predicciones, métricas, informe y manifiesto. |

El cálculo retrospectivo usado quedó asociado al commit `1e419ad881db1cc9798685ef19f414862461674b`, Python 3.13.18, NumPy 2.1.3, SciPy 1.16.3, joblib 1.6.0, threadpoolctl 3.7.0 y scikit-learn 1.9.1. El wheel de scikit-learn se verificó por SHA; para las demás dependencias se verificó la versión.

## Protección temporal

Los modelos y configuraciones se congelaron antes de la evaluación. Tras la autorización y las verificaciones previas se leyeron los bytes de Dataset v1 en Windows para calcular su SHA, antes del journal del evaluador en Kaggle; esa lectura no inspeccionó filas ni se utilizó para entrenar o seleccionar modelos. El journal registra el acceso de la ejecución retrospectiva en Kaggle, no ese hash previo.

La reproducción completa requiere conservar los artefactos originales y respetar el procedimiento de ejecución, las identidades y las restricciones descritas en [ejecución V2](et0_v2_execution.md). Falta documentar y verificar de forma aislada la carga y predicción de modelos representativos antes de cerrar este apartado.

