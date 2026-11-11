# Handoff Quality IB09 V2: snapshot pendiente

`handoff.json` es un handoff versionado con `decision_effective=false`, `official_period_verification_status=UNRESOLVED` y `v2_scientific_admission_status=PENDING`. Las fechas planas `official_start_date` y `official_end_date` son los límites del catálogo auditado; `official_period_verification.start/end/evidence` permanecen nulos. La inclusión de ambos extremos es la regla Quality del proyecto, no una confirmación independiente de la semántica del servicio oficial.

El `source_repo_sha=97aa8cc0062de5ba6d2ec1af7668495a352c3777` identifica el commit de #66 que contiene los datos y Quality POST, no el commit que publica este handoff. El `audit_source_repo_sha` es el registrado por POST. El informe y el inventario de #66 están en `data_quality/siar/ib09-v2-adjudication.md` y `data_quality/siar/ib09_v2_inventory.json`.

## Evidencia oficial consultada

El 11/11/2026 se buscó en el [portal oficial de consulta SiAR](https://servicio.mapa.gob.es/siarweb/consultaDatos/inicio): enumera IB09 Artá como inactiva, sin acreditar en la información consultada fecha de instalación, fecha de baja ni inclusión de extremos. Se buscaron adicionalmente ambas fechas junto al código IB09 en fuentes públicas, sin encontrar respuesta oficial verificable. El endpoint `Info/ESTACIONES` usado por SIAR Sync exige autenticación; en este checkout no se ha localizado una respuesta original preservada. El POST y su revisión independiente certifican el periodo del catálogo en 2026, pero no publican la respuesta individual reproducible. `official_source_sha256` es null: no se ha preservado copia oficial que pruebe las fechas. No se infiere una causa horaria. CATALOG-TZ-001 continúa abierto.

## Validación

Desde la raíz del repositorio: `python data_quality/siar/ib09-handoff-v2/validate_handoff.py .`. Devuelve válido para conservación, pero `--for-admission` falla siempre. El esquema se bloquea al snapshot exacto pendiente; una autorización o evidencia futura requiere una nueva versión con esquema y validación revisados. El validador comprueba el hash normalizado LF del esquema, los hashes de raw, catálogo, ledger, paquete POST y prototipo, y reconcilia fechas, duplicados y huecos. El contrato JSON Schema 2020-12 describe este snapshot y el validador implementa los elementos utilizados: const, type object, required, properties y additionalProperties.

05 debe validar por su cuenta los bytes y vincular cada fila del dataset V2 con el raw, preservar el FAIL V1, comprobar evidencia oficial y autorización, usar un dataset_id V2, incorporar política y decisión al linaje, y aplicar filtros específicos por estación, Quality y experimento. Este validador no acredita los valores de filas derivadas ni constituye por sí solo un gate de admisión; el prototipo de #66 permanece desconectado. La admisión requiere una decisión futura verificable y un contrato V2 de 05.

El hash del paquete POST corresponde a los bytes de Git (`a2d1293fcc54254769dbc80c7105b08b4b03b6a14392df0a4e5ef19b4ef51d07`). Una copia de trabajo Windows que transforme LF a CRLF puede fallar la comprobación; debe emplearse una extracción que preserve los bytes del commit. No se normalizan los datos de las fuentes para hacer pasar el hash.
