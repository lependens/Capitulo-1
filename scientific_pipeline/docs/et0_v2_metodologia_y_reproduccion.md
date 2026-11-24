# ET₀ V2: metodología y reproducción

> Documento preparado para revisión de Dirección. La ejecución completa no se ha repetido; las comprobaciones reproducibles y sus límites se detallan en la evidencia de certificación.

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

La reproducción completa requiere conservar los artefactos originales y respetar el procedimiento de ejecución, las identidades y las restricciones descritas en [ejecución V2](et0_v2_execution.md).

## Reproducción de la evaluación

Usar una copia de trabajo nueva del árbol congelado y comprobar el commit, la cápsula, Dataset v1 y el SHA del freeze antes de ejecutar. El original y sus modelos deben permanecer de solo lectura. Con un entorno compatible y las rutas locales configuradas como `CAPSULE`, `RUN`, `DATASET_V1` y `OUTPUT`, desde la raíz del checkout:

```bash
export PYTHONPATH="$PWD/scientific_pipeline/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m siar_science.et0_v2_retrospective --repo . --capsule "$CAPSULE" --run "$RUN" \
  --dataset "$DATASET_V1" --freeze-sha256 97acad7ce7cc39c65d82f6efdc36290ed0325caaef2dedb69d9ffbefe1addfbd --output "$OUTPUT"
```

La evaluación original es la versión privada Kaggle `370251881`. El manifiesto de salida SHA256 es `f771dc3564270528c203d9af709b2b88649a552ef7dd831bf4503d04a2fd8daa`. Comprobar cada salida con ese manifiesto; los metadatos de tiempo impiden asumir que ejecuciones posteriores produzcan necesariamente un manifiesto byte a byte idéntico.

`et0_v2_retrospective._frozen_predictions` recibe filas elegibles y la celda congelada: aplica el coeficiente AHC, invoca `predict` para ML y aplica el escalador seguido de la media de las tres redes para ANN. `retrospective_metrics` agrupa por estación y calendario y calcula error sobre las claves comunes a los cinco métodos principales. `metrics.metrics` devuelve n, MAE, RMSE, sesgo y R² predictivo; la cobertura propia se contabiliza separadamente.

La prueba aislada de carga e inferencia para Elastic Net, ANN A y ANN E sobre IB01/HG-Rs/annual reproduce exactamente las salidas archivadas; véase el [contrato de inferencia](et0_v2_contrato_inferencia.md). No verifica todas las celdas. La evaluación no vuelve a seleccionar ni ajustar modelos utilizando TEST. `et0_v2_retrospective._history` calcula por separado el AHC del histórico completo; ese ajuste incluye el periodo retrospectivo y su error es exclusivamente in-sample.

## Evidencia y límites aún abiertos

El ZIP original del lote final y los lotes intermedios se conservan por separado; se conserva el catálogo web anterior DRAFT y existe un catálogo técnico definitivo VERIFIED, con publicación NOT_PUBLISHED. La comparación global y 9.612 filas de métricas se contrastaron contra las predicciones. La restauración independiente del paquete web pasó: snapshot `backup_20261116T163815Z_35960ca0`, ZIP restaurado SHA256 `c502530fa70321246c26ed0c9c59aaf766d4ac619251bc1ab99f854695701690`. La documentación histórica del TFG no debe tratarse como si describiese esta ejecución V2. La cadena de hashes y la trazabilidad por lote de los 4.935 checkpoints se contrastaron; las 9.612 filas de métricas y sus cohortes se verificaron desde predicciones. La inferencia independiente sigue siendo una prueba puntual, no una repetición de todas las celdas.

