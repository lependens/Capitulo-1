# Estado de sincronización SIAR — Baleares

## Corrección de ledger POST — COMPLETE

El POST original SIAR-QA-POST-20261018T233901Z está preservado y superseded por SIAR-QA-POST-20261020T193140Z. Se corrigió el generador para retener las filas del ledger de Recovery persistido aunque la fecha RECOVERED ya no figure entre las ausencias del auditor. El estado de auditoría y sus métricas permanecen: Audit completeness COMPLETE; observacional 96.987410 %; evidencia sin resolver 0; RECOVERED 1; CONFIRMED_NO_DATA 2,345; PENDING_RETRY 0. Ver [informe de corrección](../../data_quality/siar/ledger-correction.md).

**Actualizado:** 2026-10-23\
**Componente observacional:** v2.3.3; **SIAR-SYNC-RUNTIME = HOLD**
**Backfill temporal:** alcanzado para las 12 estaciones
**Quality Audit v1 PRE:** COMPLETADO — válido

## Resultado histórico de Quality Audit v1 PRE

| Métrica | Resultado |
|---|---:|
| Estaciones auditadas | 12 |
| ACTIVE | 9 |
| CLOSED | 3 |
| WARN | 11 |
| FAIL | 1 (IB09) |
| INCOMPLETE | 0 |
| Días esperados | 77.840 |
| Días con dato | 75.494 |
| Huecos PRE | 2.346 |
| Cobertura ponderada | 96,99 % |

En el snapshot PRE, los 2.346 huecos quedaron registrados como `PENDING_RETRY`, `NOT_ATTEMPTED`, con `attempt_count=0`. El PRE no declaró ningún `CONFIRMED_NO_DATA`; esos estados son inmutables y no describen necesariamente la recuperación actual.

## Quality Audit POST — COMPLETE

El POST vigente es SIAR-QA-POST-20261020T193140Z; supersedes a SIAR-QA-POST-20261018T233901Z, histórico preservado. El snapshot auditado corresponde a audit_source_repo_sha=18e6d893d119d7a6bd2959d6fdeb69fbdfa2c865.

| Métrica | Resultado |
|---|---:|
| Audit completeness | COMPLETE |
| Días esperados | 77.840 |
| Días observados | 75.495 |
| Cobertura observacional | 96.987410 % |
| RECOVERED | 1 |
| CONFIRMED_NO_DATA | 2.345 |
| PENDING_RETRY | 0 |
| Unresolved evidence | 0 |
| WARN | 11 |
| FAIL | 1 (IB09, fila fuera del periodo oficial) |
| INCOMPLETE | 0 |

COMPLETE_BUT_INCOMPLETE_DATASET indica que la auditoría terminó y que la cobertura observacional es inferior al 100 % por fechas que SIAR confirmó sin registros. No significa que la auditoría esté incompleta. CONFIRMED_NO_DATA es terminal y válido para este cierre.

Los [artefactos vigentes](../../data_quality/siar/post/SIAR-QA-POST-20261020T193140Z/) contienen 2.346 filas ledger: 1 RECOVERED, 2.345 CONFIRMED_NO_DATA, 0 PENDING_RETRY. Los [originales superseded](../../data_quality/siar/post/SIAR-QA-POST-20261018T233901Z/) permanecen byte-preserved. Raw: 75.496 filas, 75.495 oficiales y 1 OUTSIDE_OFFICIAL_PERIOD.
## IB07 — recuperación incluida en POST

Quality POST clasifica IB07 / 2012-11-13 como RECOVERED. El CSV completo preservado del snapshot Quality tiene SHA-256 8641cdf90697064fe07c505618d5a9921cf103e5fc1e1d06b28812b566786c68, igual a canonical_sha256 del manifest. Conserva 2.946 filas, 23 columnas y una sola fila para esa fecha. La publicación original incorporó el CSV completo validado; la corrección del ledger no reconstruyó la fila ni volvió a consultar SIAR. FORENSIC-IB07-001/F2 verificó 23/23 campos mediante comparación canónica.

La diferencia de contenido respecto al main de origen consiste únicamente en añadir la observación validada del 2012-11-13. La procedencia de la escritura original quedó sin resolver; una consulta independiente posterior confirmó que el contenido de la fila coincide con SIAR. La política de Recovery registró esta evidencia sin alterar otras observaciones.

## Hallazgo IB09

El periodo oficial de IB09 comienza el 2005-10-24 y termina el 2018-10-10. La fila raw 2005-10-23 se conserva intacta y se clasifica científicamente como OUTSIDE_OFFICIAL_PERIOD; queda excluida del periodo científico oficial. El hallazgo produce el FAIL estructural de IB09 en Quality POST y no bloquea Dataset Observational v1 según la decisión de Dirección 00.

## CATALOG-TZ-001

CATALOG-TZ-001 = OPEN / NON_BLOCKING. El catálogo local informa Huso 31, mientras Info/ESTACIONES informa Huso 30 para 12/12 estaciones. Es un hallazgo semántico abierto y no bloqueante. No se modifica estaciones_baleares.csv ni se deriva una corrección científica durante esta publicación.

## Saneamiento operativo de permisos posterior a v2.3.2

El modo esperado de los CSV canónicos es `0644`. La lógica correctora forma parte de SIAR Sync desde v2.3.2; el saneamiento puntual de archivos existentes fue operativo, posterior a v2.3.2, y no constituye una nueva versión funcional.

Resultado verificado:

- **12/12 CSV en modo `0644`**;
- **SHA-256 PRE = POST en 12/12**;
- contenido raw sin cambios.

Estas verificaciones corresponden al saneamiento puntual de permisos: no se reescribieron filas, esquema, fechas ni observaciones durante esa operación. No describen los posibles cambios de contenido producidos posteriormente por Recovery.

## Estado de adquisición — HOLD y snapshot científico separado

Quality COMPLETE; observaciones INCOMPLETE por ausencias confirmadas. Publicación canónica source_repo_sha = `2c4874c73b6f818793cd2e0af42eeeb4cea9a6e9`; audit_source_repo_sha = `18e6d893d119d7a6bd2959d6fdeb69fbdfa2c865`.

Dataset Observational v1 ya está construido y validado PASS, SHA256 `0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb`. La primera ejecución oficial PM/HS también está PASS, publicada en PR #31; [ubicaciones y resultados](../DATA_LOCATIONS.md). No se confunde ese cierre con la reactivación de adquisición 2026.

**SIAR-SYNC-RUNTIME = HOLD** según diagnóstico de 02 comunicado por Dirección 00. Checkout residual `quality/recovery-20261012T085131Z`, HEAD `18e6d893...`; main canónico ya contiene la recuperación. Target default 2025-12-31; no se ha demostrado modo incremental 2026 sin reabrir huecos históricos. /start podría intentar aproximadamente 1.958 huecos de activas. El panel mezcla historical gaps con lag actual y no aplica correctamente Fecha_Baja a cerradas aunque el worker sí.

Requiere **SIAR-SYNC-PRODUCTION-ALIGNMENT + SIAR-SYNC-INCREMENTAL-2026** antes de levantar HOLD. El snapshot científico 2025 no depende de ello. Forecast SIAR/AEMET son independientes del runtime Sync; sus gates operativos constan en [PROJECT_STATUS](../PROJECT_STATUS.md).

Las reglas normativas permanecen en [Quality Audit v1](quality.md). No se modifican raw, ledger ni snapshots desde esta actualización documental.
