# Quality Audit v1 — datos observacionales SIAR

## Objetivo y alcance

Definir una auditoría reproducible de los 12 CSV canónicos de Baleares:

`IB01`–`IB11` y `IB101`, almacenados en `datos_siar_baleares/`.

El backfill temporal de adquisición ha alcanzado las 12 estaciones, pero la certificación científica de calidad y cobertura está pendiente. Esta especificación no declara los datos completos ni válidos. No modifica los CSV raw, no inicia una segunda pasada de adquisición y no genera resultados de auditoría por sí sola.

El catálogo raíz `estaciones_baleares.csv` es la fuente para códigos, estado de la estación y fechas `Fecha_Instalacion` / `Fecha_Baja`. El esquema raw SIAR de 23 columnas permanece intacto.

## Política de calidad

1. **Los CSV raw SIAR no se interpolan ni imputan.** Los datos faltantes se registran como hallazgos; cualquier dataset derivado deberá ser una capa separada y explícitamente aprobada.
2. **Un hueco no equivale a un error de consulta.** La ausencia de una fila en el CSV no demuestra que SIAR no tuviera datos ni que la consulta fallara.
3. **`CONFIRMED_NO_DATA` exige evidencia positiva:** una respuesta SIAR válida, interpretada correctamente, que devuelve cero registros para la estación y fecha consultadas.
4. Un timeout, error HTTP/API, respuesta inválida o consulta no ejecutada debe tener una clasificación distinta de `CONFIRMED_NO_DATA`.
5. Para una estación cerrada, el límite final de la ventana esperada es `Fecha_Baja`; no se extiende la expectativa más allá de esa fecha. La convención de inclusión diaria debe seguir la semántica confirmada del catálogo y quedar documentada en el informe.
6. Para una estación activa, la auditoría debe recibir un cutoff explícito. Se registra en el informe y se usa como límite de la ventana auditada; no se infiere automáticamente de la última fila del CSV.
7. Los resultados deben conservar trazabilidad: identificador de auditoría, fechas UTC, rutas, SHA-256 y evidencia de consulta cuando se clasifique un día.
8. Un resultado desconocido permanece desconocido. No convertir vacíos o errores en ceros.

## Formato común de los informes CSV

Los tres informes son artefactos de salida de una ejecución futura, no parte de los CSV raw:

- UTF-8 sin BOM, delimitador coma, una fila de encabezado;
- fechas como `YYYY-MM-DD` y timestamps UTC ISO 8601 terminados en `Z`;
- booleanos `true` / `false`;
- números enteros decimales sin separadores de millares;
- campos desconocidos o no aplicables vacíos, nunca reemplazados por cero;
- orden estable de columnas según los encabezados especificados abajo.

Los archivos de informe todavía **no se han generado**. No añadir filas de ejemplo o snapshots inventados al repositorio.

## 1. `siar_station_audit.csv`

Una fila por estación y ejecución de auditoría. Encabezado:

```csv
audit_id,audited_at_utc,station_code,station_name,station_state,fecha_instalacion,fecha_baja,active_cutoff_date,expected_start_date,expected_end_date,canonical_file,canonical_sha256,row_count,first_observation_date,last_observation_date,duplicate_date_count,invalid_row_count,missing_day_count,audit_status,notes
```

- `station_state`: `ACTIVE` o `CLOSED`, contrastado con el catálogo.
- `fecha_instalacion` y `fecha_baja`: fechas del catálogo; `fecha_baja` queda vacía para estaciones activas si así figura en la fuente.
- `active_cutoff_date`: obligatorio para una estación activa; vacío para una cerrada.
- `expected_start_date` / `expected_end_date`: límites de la ventana realmente evaluada, sin extrapolarlos.
- `canonical_file`: ruta relativa al repositorio; `canonical_sha256`: hash del archivo auditado.
- `audit_status`: `PASS`, `WARN`, `FAIL`, `INCOMPLETE` o `NOT_RUN`. `NOT_RUN` sirve solo para un plan, no para declarar certificación.

## 2. `siar_missing_days.csv`

Una fila por fecha candidata a hueco o incidencia, no una fila ficticia por cada día aún no auditado. Encabezado:

```csv
audit_id,station_code,observation_date,classification,query_attempted,siar_response_valid,siar_records_returned,error_code,evidence_ref,checked_at_utc,notes
```

Valores permitidos de `classification`:

- `CONFIRMED_NO_DATA`: consulta realizada y respuesta SIAR válida con cero registros.
- `QUERY_ERROR`: la consulta se intentó, pero falló por transporte, HTTP/API u otro error registrado.
- `INVALID_RESPONSE`: hubo respuesta, pero no pudo validarse o interpretarse con seguridad.
- `NOT_QUERIED`: no existe una consulta comprobatoria para esa fecha.
- `DATASET_GAP`: SIAR devolvió registros válidos para la fecha, pero el CSV canónico no los contiene.

Para `CONFIRMED_NO_DATA`, `query_attempted=true`, `siar_response_valid=true` y `siar_records_returned=0` son obligatorios. Un error de consulta jamás se etiqueta como ausencia confirmada.

## 3. `siar_dataset_manifest.csv`

Una fila por CSV canónico incluido en el snapshot de auditoría. Encabezado:

```csv
audit_id,manifest_created_at_utc,station_code,canonical_file,canonical_sha256,file_size_bytes,row_count,first_observation_date,last_observation_date,expected_start_date,expected_end_date,duplicate_date_count,invalid_row_count,missing_day_count,audit_status
```

Los valores describen los bytes efectivamente auditados. El manifiesto no modifica, normaliza ni copia el CSV. Su SHA-256 permite detectar cambios entre auditorías.

## Estructura y ejecución

Los destinos reservados están bajo `data_quality/siar/`. Su README identifica los informes esperados. La generación de archivos y cualquier segunda pasada de adquisición requieren una decisión posterior; esta PR solo documenta política, formatos y ubicación.
