# Informe consolidado — POST superseding de Quality

**Estado:** QUALITY-POST-RECOVERED-LEDGER-CORRECTION = PASS  
**Audit ID:** SIAR-QA-POST-20261020T193140Z  
**Audit time UTC:** 2026-10-20T19:40:51Z  
**Workspace:** /home/josep/quality-audit-post/SIAR-QA-POST-20261020T193140Z  
**Audit source repo SHA:** 18e6d893d119d7a6bd2959d6fdeb69fbdfa2c865  
**PRE ID / repo SHA:** SIAR-QA-PRE-20261012T064810Z / e850827e24cff11ae46f51c9bc3e8a2f36b0d2da

Este POST supersede el artefacto publicado del SIAR-QA-POST-20261018T233901Z porque aquel omitió la fila ledger IB07/2012-11-13. El PRE y el POST original permanecen intactos en sus workspaces. No se consultó SIAR, no se reinició R2 y no se modificó ningún CSV raw.

## Dictamen de calidad

- Audit completeness: **COMPLETE**.
- Observational completeness: **INCOMPLETE — 96.987410%**.
- Unresolved evidence: **0**.
- Expected station-days: **77,840**.
- Observed station-days: **75,495**.
- Coverage: **96.987410%**.
- Gap ledger: **RECOVERED=1; CONFIRMED_NO_DATA=2,345; PENDING_RETRY=0**.
- Unobserved dates: **2,345**. The ledger has **2,346** rows because it also retains the recovered date.
- Station audit statuses: **PASS=0; WARN=11; FAIL=1; INCOMPLETE=0**.
- Governance outcome remains **COMPLETE_BUT_INCOMPLETE_DATASET**. This means the audit is complete while the observation series is below 100%; it does not mean the audit itself is incomplete.

## Estado por estación

| Station | State | Expected | Observed | Coverage | RECOVERED | CONFIRMED_NO_DATA | PENDING_RETRY | Outside period | Audit status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| IB01 | ACTIVE | 7,947 | 7,716 | 97.09% | 0 | 231 | 0 | 0 | WARN |
| IB02 | ACTIVE | 7,946 | 7,697 | 96.87% | 0 | 249 | 0 | 0 | WARN |
| IB03 | ACTIVE | 7,945 | 7,873 | 99.09% | 0 | 72 | 0 | 0 | WARN |
| IB04 | ACTIVE | 7,945 | 7,729 | 97.28% | 0 | 216 | 0 | 0 | WARN |
| IB05 | ACTIVE | 7,941 | 7,834 | 98.65% | 0 | 107 | 0 | 0 | WARN |
| IB06 | ACTIVE | 7,941 | 7,862 | 99.01% | 0 | 79 | 0 | 0 | WARN |
| IB07 | CLOSED | 3,017 | 2,946 | 97.65% | 1 | 71 | 0 | 0 | WARN |
| IB08 | ACTIVE | 7,393 | 7,123 | 96.35% | 0 | 270 | 0 | 0 | WARN |
| IB09 | CLOSED | 4,735 | 4,422 | 93.39% | 0 | 313 | 0 | 1 | FAIL |
| IB10 | ACTIVE | 7,390 | 7,007 | 94.82% | 0 | 383 | 0 | 0 | WARN |
| IB101 | CLOSED | 252 | 249 | 98.81% | 0 | 3 | 0 | 0 | WARN |
| IB11 | ACTIVE | 7,388 | 7,037 | 95.25% | 0 | 351 | 0 | 0 | WARN |

## Fila ledger recuperada

La fila se copió del estado persistido siar_missing_days_working.csv y del objeto gap_after de RECONCILE_IB07_applied_20261013T054128Z.json. Solo audit_id y repo_git_sha se actualizaron al contexto de este POST; los demás campos coinciden exactamente con la evidencia persistida.

| Field | Persisted value |
|---|---|
| audit_id | SIAR-QA-POST-20261020T193140Z |
| repo_git_sha | 18e6d893d119d7a6bd2959d6fdeb69fbdfa2c865 |
| siar_sync_version | 2.3.3 |
| audit_tool_version | 1.0.0 |
| estaciones_catalog_sha256 | 98abd16dcd6ea42ad46a30c331df47150f07dffdca140c14e4445890ce68a357 |
| quality_spec_version | 1 |
| station_code | IB07 |
| observation_date | 2012-11-13 |
| gap_status | RECOVERED |
| last_query_result | DATA_RETURNED |
| first_detected_at_utc | 2026-10-12T07:37:40Z |
| last_attempt_at_utc | (vacío) |
| attempt_count | 0 |
| recovered_at_utc | 2026-10-12T21:22:05.511553+00:00 |
| siar_records_returned | 1 |
| evidence_ref | FORENSIC-IB07-001/F2 |
| error_code | (vacío) |
| notes | PRE gap detected from canonical CSV; no SIAR query attempted |

### Evidencia F2

- FORENSIC-IB07-001/F2: un endpoint de datos, HTTP 200, una fila devuelta; payload SHA-256 f475594bb1605badede971b4062edc2c28682048ecfe8cccb36b327021d3343a.
- La comparación canónica independiente acredita **23/23 campos**, content_validity=DEMONSTRATED, y la fila raw está presente exactamente una vez.
- El comparador bruto de la consulta registró una diferencia sensible a mayúsculas para HumedadMin/humedadMin; la comparación canónica posterior aplicó la normalización de nombre de campo y demostró la igualdad 23/23. No hay discrepancia de valores meteorológicos.
- original_write_provenance permanece UNRESOLVED; content_confirmed_by_independent_query=TRUE.
- SHA de archivos de evidencia: {"F2_query_evidence_sha256": "fd78beb498c7f274a786b45428f537b13a23f1ac67de23fdb00306ad41972039", "F2_canonical_comparison_sha256": "7aace27f1d26ba3494275a66e45d84faa652e055c8869e5773bc8dee1c7cc406", "RECONCILE_applied_json_sha256": "0c8893e1a94972066f8bc2c4b86a4e4a30e0daaa6b469a4a1915546670ed45b4", "persisted_recovery_ledger_sha256": "8c91648b52be694215812869a36657ca208cd0ecbbef9a6d8c6340b8937dd243"}.

## Defecto y corrección del generador

El build_post.py original usaba las filas que el auditor clasificaba como ausentes como lista final del ledger. Luego mezclaba el estado de Recovery solo en esas filas. Por eso la fila recuperada, aunque persistía en R2 y estaba validada contra raw, quedaba excluida del CSV siar_missing_days.csv. El mismo orden también re-fechaba first_detected_at_utc con la hora de cada auditoría en vez de conservar el dato persistido.

La versión corregida usa la lista del auditor únicamente para validar qué claves siguen sin observación y construye el ledger de salida desde el estado persistido de Recovery. Conserva los campos históricos, evita duplicados y exige que las claves no observadas coincidan exactamente con PENDING_RETRY y CONFIRMED_NO_DATA. Se ejecutaron **3 pruebas unitarias**, todas PASS.

El POST corregido preserva los valores first_detected_at_utc persistidos para las 2.346 filas. Frente al POST anterior, las 2.345 filas ya publicadas conservan su estado y el resto de trazabilidad; su timestamp de primera detección se restaura al valor del ledger persistido. La única clave añadida es IB07/2012-11-13.

## Comparación PRE → POST

| Métrica | PRE | POST superseding |
|---|---:|---:|
| Expected station-days | 77,840 | 77,840 |
| Observed station-days | 75,494 | 75,495 |
| Coverage | 96.986125% | 96.987410% |
| Raw rows | 75,495 | 75,496 |
| Unobserved dates | 2,346 | 2,345 |
| Gap ledger rows | 2,346 | 2,346 |
| Gap classification | 2,346 PENDING_RETRY | 1 RECOVERED; 2,345 CONFIRMED_NO_DATA; 0 PENDING_RETRY |

La cobertura aumenta por la única observación real de IB07 ya presente y validada en raw; los días CONFIRMED_NO_DATA continúan sin aumentar la cobertura.

## Raw, PRE y hallazgos

- Los 12 CSV del repo SHA 18e6d893d119d7a6bd2959d6fdeb69fbdfa2c865 coinciden con canonical_sha256 del manifest; cada archivo conserva 23 columnas canónicas.
- IB07 raw SHA-256: 8641cdf90697064fe07c505618d5a9921cf103e5fc1e1d06b28812b566786c68. La observación 2012-11-13 coincide con F2 y aparece una vez.
- IB09 raw SHA-256: 2cbd9bb289404264fc78dad21322562d9d22b64ce4bbc87e51512ca763ea7a8a. IB09/2005-10-23 sigue intacta; se mantiene rows_outside_official_period_count=1, FAIL estructural y política científica OUTSIDE_OFFICIAL_PERIOD.
- CATALOG-TZ-001 permanece **OPEN / NON_BLOCKING**: Huso local 31 frente a Info/ESTACIONES 30 en 12/12 estaciones. No se modificó estaciones_baleares.csv.
- PRE y POST original: hashes aceptados verificados contra sus registros y bytes, sin sobrescritura.
- Repo de datos: branch quality/recovery-20261012T085131Z, HEAD 18e6d893d119d7a6bd2959d6fdeb69fbdfa2c865, working tree limpio. El generador y las herramientas de revisión están solo en este workspace superseding.

## SHA-256 del POST superseding

- siar_station_audit.csv: e444cebf4165939c24cb0e9c132b241363bb60b0808e4403e1cf2ac33bd6be7d
- siar_missing_days.csv: 55a78e6affff09e2a1e8cb28e0465b2073ad79a1e573a1efe272a52462ba3969
- siar_dataset_manifest.csv: b37c6ef2cf47f4ac7ac7308d0414b39429c5a35c352ec09581f9f4751468337b
- post_report.md: 383317b2635e720533b89250b6f5bac3cd1a7e1b17f47ed318246afeeedfe1dd

El informe consolidado y la certificación independiente tienen sus hashes en superseding_checksums.sha256, que también incluye los tres CSV anteriores.

## Revisión independiente y certificación

El verificador reproducible verify_superseding_post.py pasó **2,438 assertions**, incluidos los 12 hashes de raw, esquemas, todos los campos del ledger comparados con el ledger persistido, F2 23/23, métricas, IB09, PRE inmutable y working tree.

**Certificación:** PASS. No se inicia recovery, no hay nuevas consultas SIAR, no se altera raw y Dataset Baleares v1 no se declara cerrado.
