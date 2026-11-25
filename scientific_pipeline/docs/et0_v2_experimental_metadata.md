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


## Publicación verificada para el handoff web (2026-11-25)

La exportación de esta versión quedó instalada y verificada en:

`/srv/siar-project/scientific-artifacts/et0-v2-experimental-metadata-v1/`

El catálogo complementario `catalog.json` tiene SHA256 `3207fe042abb004b7cd69122339845adc8d55cd65b5cfd4d676d9e5512e6eea9`. Conserva las identidades de Dataset v1 (`0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb`) y freeze (`97acad7ce7cc39c65d82f6efdc36290ed0325caaef2dedb69d9ffbefe1addfbd`). El inventario firmado de los 11 archivos tiene SHA256 `ffd32445e07c596434e75e80fa7a23e939ded475bbfae3e284178c55cc9ec947`; su recibo de ingestión está en `/srv/siar-project/ingestion-receipts/science-20261125T122837Z-29599217_c2839b84.json` y registra `published-verified`.

Se cotejaron tamaños y SHA256 de las diez tablas frente al catálogo tanto en la ubicación publicada como en el snapshot remoto. La copia cifrada quedó `REMOTE_VERIFIED` bajo el backup `backup_20261125T123805Z_0944177f`; el SHA256 de su manifiesto es `ca1449888e2986a202fc60faf6902619d7f8387181efdb64a48d89a62cc3686b`. La restauración independiente se realizó en la cuarentena temporal `/srv/siar-project/preservation/restores/backup_20261125T123805Z_0944177f_8513c71d/`. El procedimiento devolvió `RESTORE_HASHES_PASS ... files=46`; dentro de la restauración se verificó el conjunto exacto de 11 archivos: el catálogo conserva SHA256 `3207fe042abb004b7cd69122339845adc8d55cd65b5cfd4d676d9e5512e6eea9` y los diez hashes de tabla coinciden con `catalog.json`.

### Contrato disponible para 06

- Los diez CSV son UTF-8; los esquemas de columnas, conteos de filas, tamaños y hashes están declarados en `catalog.json`. No se publicaron binarios de modelos.
- La clave de una celda experimental es `station_code, family, calendar`; `cell_id` es su identificador compuesto. Candidatos, folds y semillas agregan `stage`, `candidate_id`, `year`, `seed_seed` o `group` según el grano indicado arriba.
- El inventario declara disponibilidad y falta de soporte por celda. No se deben completar las tres celdas mensuales de IB101 que carecen de soporte ni inferir métricas o predicciones para ellas.
- Para la unión con resultados retrospectivos, comprobar las claves de estación, método/familia, resolución/calendario y fecha, además de elegibilidad y disponibilidad de referencia; la tabla de candidatos DEVELOPMENT no representa predicciones diarias ni resultados TEST.
- `ahc-history-in-sample.csv` es exclusivamente in-sample. Las selecciones y métricas de candidatos son DEVELOPMENT; las métricas retrospectivas permanecen en el catálogo existente. No se cambió el estado científico `NOT_PUBLISHED`.

Este paquete está instalado y listo para que 06 lo consuma en la integración autorizada. La carga en la aplicación no implica publicación científica ni altera el catálogo retrospectivo firmado.