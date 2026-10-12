# Quality Audit v1 — datos observacionales SIAR

## Objetivo y alcance

Definir una auditoría reproducible de los 12 CSV canónicos de Baleares: IB01–IB11 y IB101, almacenados en datos_siar_baleares/.

El backfill temporal de adquisición ha alcanzado las 12 estaciones, pero la certificación científica de calidad y cobertura sigue pendiente. Esta especificación no declara los datos completos ni válidos. No modifica los CSV raw, no inicia una segunda pasada de adquisición y no genera resultados de auditoría por sí sola.

## Fuentes y autoridad del catálogo

estaciones_baleares.csv en la raíz del repositorio es un snapshot local reproducible y versionado, útil para repetir auditorías y registrar su SHA-256. No es por sí solo la autoridad vigente sobre el catálogo SIAR.

Antes de certificar, contrastar códigos, estado y fechas de instalación/baja con Info/ESTACIONES de SIAR, que es la autoridad a consultar. Registrar el snapshot local empleado y su hash; documentar discrepancias y no certificar estaciones afectadas hasta resolverlas.

El esquema raw SIAR esperado es de 23 columnas. No se modifica.

## Política de calidad

1. **Nunca interpolar ni imputar los CSV raw SIAR.** Los datos faltantes se registran como hallazgos; cualquier dataset derivado requiere una capa separada y aprobación explícita.
2. **Un hueco no equivale a un error de consulta.** La ausencia de una fila en el CSV no demuestra que SIAR no tuviera datos ni que la consulta fallara.
3. **CONFIRMED_NO_DATA exige evidencia positiva:** una respuesta SIAR válida, interpretada correctamente, que devuelve cero registros para la estación y fecha consultadas.
4. Un error HTTP, cuota, respuesta inválida o consulta no ejecutada nunca se etiqueta como ausencia confirmada.
5. Las estaciones cerradas tienen como fin de periodo oficial Fecha_Baja. Las activas usan un cutoff explícito de auditoría. Los límites inclusivos deben quedar definidos de acuerdo con la semántica del catálogo y aplicarse consistentemente.
6. Una fecha fuera del periodo oficial de la estación no constituye un hueco y no se añade a siar_missing_days.csv.
7. Un resultado desconocido permanece desconocido; no convertir vacíos o errores en ceros.

## Trazabilidad común

Cada ejecución debe registrar estos campos en los informes de auditoría/manifiesto:

- repo_git_sha: commit exacto del repositorio auditado;
- siar_sync_version: versión de SIAR Sync asociada a los CSV;
- audit_tool_version: versión del ejecutable/herramienta de auditoría;
- estaciones_catalog_sha256: SHA-256 del snapshot local estaciones_baleares.csv;
- quality_spec_version: versión de esta especificación.

Los valores deben identificar el estado realmente usado; no se rellenan con valores supuestos.

## Formato común de los informes CSV

Los tres informes son artefactos de salida de una ejecución futura, no parte de los CSV raw:

- UTF-8 sin BOM, delimitador coma y una fila de encabezado;
- fechas como YYYY-MM-DD y timestamps UTC ISO 8601 terminados en Z;
- booleanos true / false;
- números enteros decimales sin separadores de millares;
- campos desconocidos o no aplicables vacíos, nunca reemplazados por cero;
- orden estable de columnas según los encabezados especificados abajo.

Quality Audit v1 PRE ya se ejecutó. Los informes y snapshots de esa ejecución son artefactos separados: esta actualización documental no los genera, copia ni modifica. No añadir filas de ejemplo ni snapshots inventados.

## 1. siar_station_audit.csv

Una fila por estación y ejecución. Encabezado:

```csv
audit_id,audited_at_utc,repo_git_sha,siar_sync_version,audit_tool_version,estaciones_catalog_sha256,quality_spec_version,station_code,station_name,station_state,fecha_instalacion,fecha_baja,active_cutoff_date,expected_start_date,expected_end_date,official_period_days,canonical_file,canonical_sha256,file_size_bytes,filesystem_mode,filesystem_mode_ok,permission_normalized_at_utc,row_count,column_count,schema_23_columns_ok,first_observation_date,last_observation_date,invalid_date_count,rows_outside_official_period_count,estacion_value_mismatch_count,chronological_order_ok,out_of_order_row_count,duplicate_date_count,invalid_row_count,days_with_data,coverage_percent,missing_day_count,audit_status,notes
```

Definiciones:

- station_state: ACTIVE o CLOSED, contrastado con Info/ESTACIONES.
- fecha_instalacion / fecha_baja: valores del snapshot local; registrar y resolver discrepancias con la autoridad antes de certificar.
- active_cutoff_date: obligatorio para estación activa; vacío para cerrada. expected_start_date y expected_end_date delimitan el periodo oficial auditado.
- official_period_days: número de días del periodo esperado según las fechas oficiales y la regla inclusiva documentada.
- column_count: número observado de columnas del CSV. schema_23_columns_ok solo es true si hay exactamente 23 columnas y coinciden con el esquema esperado.
- invalid_date_count: filas con fecha ausente o no interpretable. rows_outside_official_period_count: filas cuya fecha cae fuera del periodo de la estación.
- estacion_value_mismatch_count: filas en las que el valor de la columna Estacion no coincide con la estación auditada según el catálogo.
- chronological_order_ok: true si las fechas de las filas están en orden no decreciente; out_of_order_row_count cuenta las inversiones detectadas.
- days_with_data: fechas distintas dentro del periodo oficial con al menos una fila válida. Duplicados y filas fuera de periodo no aumentan este valor.
- coverage_percent: days_with_data / official_period_days * 100; informar con precisión decimal fija documentada por la implementación.
- filesystem_mode: modo observado del archivo en octal de cuatro caracteres, por ejemplo 0644. filesystem_mode_ok solo es true cuando filesystem_mode == 0644.
- permission_normalized_at_utc: timestamp verificable de la normalización operativa, si ya se realizó; vacío si no se ha realizado o no se conoce. La auditoría PRE puede registrar el modo real observado.
- audit_status: PASS, WARN, FAIL, INCOMPLETE o NOT_RUN. NOT_RUN sirve para un plan, nunca para declarar certificación.

## 2. siar_missing_days.csv

Registro persistente de fechas candidatas a hueco dentro del periodo oficial. Una fecha fuera del periodo no se registra como hueco. Encabezado:

```csv
audit_id,repo_git_sha,siar_sync_version,audit_tool_version,estaciones_catalog_sha256,quality_spec_version,station_code,observation_date,gap_status,last_query_result,first_detected_at_utc,last_attempt_at_utc,attempt_count,recovered_at_utc,siar_records_returned,evidence_ref,error_code,notes
```

Valores permitidos:

- gap_status: PENDING_RETRY, CONFIRMED_NO_DATA, RECOVERED.
- last_query_result: NOT_ATTEMPTED, ZERO_ROWS, DATA_RETURNED, HTTP_ERROR, INVALID_RESPONSE, QUOTA.

Reglas:

- Cada registro conserva first_detected_at_utc; last_attempt_at_utc se actualiza con el intento más reciente; attempt_count cuenta todos los intentos, incluido el último.
- NOT_ATTEMPTED, HTTP_ERROR, INVALID_RESPONSE y QUOTA no prueban ausencia de observaciones y mantienen gap_status=PENDING_RETRY.
- CONFIRMED_NO_DATA solo es válido con una respuesta SIAR válida y cero registros: last_query_result=ZERO_ROWS y siar_records_returned=0.
- Si la consulta devuelve datos, verificar la fecha y la respuesta. Cuando los datos están presentes en el CSV canónico, actualizar el mismo registro a gap_status=RECOVERED, last_query_result=DATA_RETURNED y completar recovered_at_utc; mantener la trazabilidad de intentos.
- No crear filas para fechas fuera del periodo oficial, ni para fechas aún no evaluadas sin detección de hueco.

## 3. siar_dataset_manifest.csv

Una fila por CSV canónico incluido en el snapshot de auditoría. Encabezado:

```csv
audit_id,manifest_created_at_utc,repo_git_sha,siar_sync_version,audit_tool_version,estaciones_catalog_sha256,quality_spec_version,station_code,canonical_file,canonical_sha256,file_size_bytes,filesystem_mode,filesystem_mode_ok,permission_normalized_at_utc,row_count,column_count,schema_23_columns_ok,first_observation_date,last_observation_date,expected_start_date,expected_end_date,official_period_days,days_with_data,coverage_percent,duplicate_date_count,invalid_date_count,rows_outside_official_period_count,estacion_value_mismatch_count,chronological_order_ok,audit_status
```

Los valores describen los bytes efectivamente auditados. El manifiesto no modifica, normaliza ni copia el CSV. Su SHA-256 permite detectar cambios entre auditorías.

## Saneamiento operativo de permisos

El modo canónico esperado de los CSV es 0644. La lógica correctora ya forma parte de SIAR Sync desde v2.3.2; normalizar archivos preexistentes es un **saneamiento operativo posterior a v2.3.2**, no una nueva versión funcional.

PR #15 solo define cómo auditar y documentar el modo. La normalización real se coordina separadamente con 01/02: hash PRE, cambio de modo a 0644, hash POST, comprobación de igualdad de hashes y verificación de que los 12 CSV quedan en 0644. No se deben reescribir los CSV ni modificar contenido, filas, esquema, fechas u observaciones. GitHub no conserva estos bits ordinarios y no puede hacer el saneamiento del filesystem.

## Estructura y ejecución

Los destinos reservados están bajo data_quality/siar/. Su README identifica los informes esperados. Quality Audit v1 PRE está completado; Quality Recovery / POST está autorizado y pendiente. La creación o actualización de resultados pertenece a una ejecución de auditoría autorizada y queda fuera de esta PR documental.
