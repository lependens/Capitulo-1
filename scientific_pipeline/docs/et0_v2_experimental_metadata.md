# ET₀ V2: metadatos experimentales DEVELOPMENT

Paquete complementario del catálogo retrospectivo firmado. Su estado de publicación es NOT_PUBLISHED. Describe selecciones y validaciones sobre DEVELOPMENT 2004–2019. La tabla de histórico completo es in-sample. Las métricas retrospectivas 2020–2025 se consultan en el catálogo existente.

El extractor verifica freeze, cápsula, vista de desarrollo, índices, fichas, checkpoints y los 945 archivos de modelo por SHA256, y verifica por separado el archivo histórico. No carga ni ejecuta los modelos. El resultado se escribe en un directorio nuevo.

```bash
python scientific_pipeline/tools/export_et0_v2_experimental_metadata.py \
  --run /ruta/al/arbol/congelado \
  --capsule /ruta/a/capsula-development \
  --history /ruta/ahc-full-history-in-sample.csv \
  --output /ruta/nueva/experimental-metadata
```

Cada CSV es UTF-8 y los campos compuestos contienen JSON. Las columnas exactas y hashes figuran en catalog.json. La clave común es `station_code, family, calendar`; `cell_id` las concatena con doble guion bajo. Las predicciones retrospectivas originales usan esas mismas tres columnas más `date`; los derivados web utilizan `station_id, method_family, resolution, date`. La unión temporal debe comprobar la disponibilidad de referencia y cohortes.

| Archivo | Grano y contenido |
|---|---|
| experimental-cells.csv | 108 claves estación-familia-calendario, soporte por grupo y fase, fechas extremas de pares elegibles y metadatos de estación. |
| ahc-frozen-coefficients.csv | 600 grupos identificados de DEVELOPMENT, coeficiente adimensional, n y suma de cuadrados. |
| ahc-fold-coefficients.csv | 8.382 grupos de folds, clave adicional year y group. |
| ahc-history-in-sample.csv | 36 registros originales de histórico completo, alcance in-sample. |
| ml-development-candidates.csv | 3.885 candidatos por celda, selección, entradas, configuración, métricas resumidas y avisos conservados. |
| ml-final-models.csv | 735 estimadores, hashes y selección principal. |
| ml-fold-metrics.csv | 55.056 registros candidato-fold. |
| ann-development-candidates.csv | 1.050 candidatos A/E, arquitectura, parámetros, entradas y resumen de folds. |
| ann-selected-architectures.csv | 210 ensembles congelados, arquitectura y épocas por semilla. |
| ann-fold-seed-metrics.csv | 44.640 registros candidato-fold-semilla, con métricas de semilla y del ensemble en columnas distintas. |

Claves adicionales: `stage` distingue ML, ANN_A y ANN_E; `candidate_id` identifica configuración dentro de la celda; `year` identifica el fold; `seed_seed` identifica semilla; `group` identifica categoría AHC. `frozen_sha256` y `checkpoint_sha256` vinculan la salida a los bytes originales. En AHC `sum_x2` es Σx² y `coefficient` es Σxy/Σx²; Σxy no se guardó como campo independiente. Las fechas de la tabla AHC identifican el intervalo contractual, no certifican observación diaria continua.

No se generan métricas TEST para candidatos descartados. El contenido de `warnings` reproduce avisos de candidato y no equivale a un contador de ajustes que no convergieron. Los modelos finales se identifican por hash, sin copiar sus binarios.

Ejemplo SQL para unir selección ANN y métricas:

```sql
SELECT a.station_code, a.family, a.calendar, a.stage, a.selected_configuration,
       m.n, m.rmse, m.bias
FROM ann_selected_architectures a
JOIN retrospective_metrics m
  ON m.group = 'station' AND m.key = a.station_code
 AND m.family = a.family AND m.calendar = a.calendar AND m.method = a.stage;
```

El catálogo de salida registra hashes por archivo, identidades de fuente y nombres de columnas. La verificación científica y preservación del paquete instalado requieren evidencia independiente. La ruta provisional de servidor debe verificarse antes de usarla.
