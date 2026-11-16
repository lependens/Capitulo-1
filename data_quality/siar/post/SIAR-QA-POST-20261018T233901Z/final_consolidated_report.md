# SIAR Quality Recovery / Quality Audit POST — Informe consolidado final 01W

## Identificación y resultado

- **Audit ID POST:** SIAR-QA-POST-20261018T233901Z
- **Audit ID PRE congelado:** SIAR-QA-PRE-20261012T064810Z
- **Fecha del audit POST:** 2026-10-18T23:39:06Z
- **Repo SHA final:** 18e6d893d119d7a6bd2959d6fdeb69fbdfa2c865
- **Branch:** quality/recovery-20261012T085131Z; working tree limpio
- **Estado global del auditor POST:** FAIL
- **Resultado de cierre certificado:** COMPLETE_BUT_INCOMPLETE_DATASET
- **Revisión independiente:** PASS_WITH_KNOWN_POST_FINDINGS, 2026-10-18T23:44:51Z

La recuperación y la auditoría POST terminaron. El auditor conserva FAIL por una violación estructural raw conocida de IB09. El conjunto mantiene 2.345 fechas sin observación confirmadas por respuestas válidas de SIAR; por ello el dataset no queda completo. No hay PENDING_RETRY ni estaciones INCOMPLETE por falta de evidencia.

## Totales y cobertura POST

- Días estación esperados: **77.840**
- Días estación observados: **75.495**
- Cobertura: **96,987410 %**
- Fechas sin observación (todas CONFIRMED_NO_DATA): **2.345**
- RECOVERED: **1**
- CONFIRMED_NO_DATA: **2.345**
- PENDING_RETRY: **0**
- Filas raw totales: **75.496**
- Filas fuera del periodo oficial: **1**

CONFIRMED_NO_DATA representa fechas verificadas individualmente con respuesta válida de SIAR y cero registros; no se suma a los días observados ni aumenta artificialmente la cobertura.

## Estado por estación

| Estación | Estado oficial | Periodo esperado inclusivo | Filas raw | Días observados | Días sin observación | Cobertura | Auditoría | Fuera de periodo |
|---|---|---|---:|---:|---:|---:|---|---:|
| IB01 | ACTIVE | 2004-03-30 a 2025-12-31 | 7716 | 7716 | 231 | 97,09 % | WARN | 0 |
| IB02 | ACTIVE | 2004-03-31 a 2025-12-31 | 7697 | 7697 | 249 | 96,87 % | WARN | 0 |
| IB03 | ACTIVE | 2004-04-01 a 2025-12-31 | 7873 | 7873 | 72 | 99,09 % | WARN | 0 |
| IB04 | ACTIVE | 2004-04-01 a 2025-12-31 | 7729 | 7729 | 216 | 97,28 % | WARN | 0 |
| IB05 | ACTIVE | 2004-04-05 a 2025-12-31 | 7834 | 7834 | 107 | 98,65 % | WARN | 0 |
| IB06 | ACTIVE | 2004-04-05 a 2025-12-31 | 7862 | 7862 | 79 | 99,01 % | WARN | 0 |
| IB07 | CLOSED | 2004-08-11 a 2012-11-13 | 2946 | 2946 | 71 | 97,65 % | WARN | 0 |
| IB08 | ACTIVE | 2005-10-05 a 2025-12-31 | 7123 | 7123 | 270 | 96,35 % | WARN | 0 |
| IB09 | CLOSED | 2005-10-24 a 2018-10-10 | 4423 | 4422 | 313 | 93,39 % | FAIL | 1 |
| IB10 | ACTIVE | 2005-10-08 a 2025-12-31 | 7007 | 7007 | 383 | 94,82 % | WARN | 0 |
| IB101 | CLOSED | 2008-12-02 a 2009-08-10 | 249 | 249 | 3 | 98,81 % | WARN | 0 |
| IB11 | ACTIVE | 2005-10-10 a 2025-12-31 | 7037 | 7037 | 351 | 95,25 % | WARN | 0 |

Distribución: **FAIL 1, WARN 11, INCOMPLETE 0, PASS 0**. Los WARN reflejan huecos confirmados sin dato y hallazgos no bloqueantes. El FAIL es IB09.

## Comparación PRE → POST

| Métrica | PRE | POST | Cambio |
|---|---:|---:|---:|
| Repo SHA | e850827e24cff11ae46f51c9bc3e8a2f36b0d2da | 18e6d893d119d7a6bd2959d6fdeb69fbdfa2c865 | POST usa HEAD final auditado |
| Filas raw | 75495 | 75496 | +1 |
| Días estación esperados | 77840 | 77840 | 0 |
| Días estación observados | 75494 | 75495 | +1 |
| Cobertura | 96,986125 % | 96,987410 % | +0,001285 puntos porcentuales |
| Fechas sin observación | 2346 | 2345 | -1 |
| RECOVERED | 0 | 1 | +1 |
| CONFIRMED_NO_DATA | 0 | 2345 | +2345 |
| PENDING_RETRY | 2346 | 0 | -2346 |
| Auditoría FAIL / WARN / INCOMPLETE / PASS | 1 / 11 / 0 / 0 | 1 / 11 / 0 / 0 | Sin cambio |
| Filas raw fuera de periodo | 1 | 1 | Sin cambio |

Los tres artefactos PRE permanecen inmutables; sus hashes registrados coinciden con la evidencia PRE.

## Hallazgos requeridos

### IB07 — FORENSIC-IB07-001 / RECONCILE-IB07-001

La fila IB07/2012-11-13 fue verificada mediante consulta SIAR independiente: coincide en sus 23 campos con el registro devuelto y está presente una sola vez en el CSV canónico. Se conservó y quedó como **RECOVERED**. La procedencia de la escritura original sigue **UNRESOLVED**; la validez del contenido está demostrada por F2. El único cambio raw respecto a la base canónica es esta incorporación validada. No se editaron otras filas raw.

### IB09

IB09 contiene una fila raw para **2005-10-23**, anterior al inicio oficial del periodo (**2005-10-24**). La fila se mantuvo intacta y cuenta como una fila fuera de periodo. Esta violación estructural es el único FAIL de estación y mantiene el resultado global del auditor en **FAIL**.

### CATALOG-TZ-001

Hallazgo pasivo: el catálogo local indica **Huso=31** y Info/ESTACIONES indica **Huso=30** en **12/12 estaciones**. No se modificó estaciones_baleares.csv ni se aplicó conversión de fechas.

## Integridad, recuperación y revisión independiente

- Esquema canónico de **23 columnas** y permisos filesystem **0644** verificados para los 12 CSV raw.
- Working tree limpio en el repo auditado; HEAD/branch corresponden al SHA indicado arriba.
- La evidencia de R2 verifica cada fecha individualmente; los 2.345 estados CONFIRMED_NO_DATA se basan en respuesta válida de cero filas. Las respuestas no concluyentes no se usaron para confirmar ausencia.
- La revisión independiente pasó las comprobaciones de hashes PRE, trazabilidad por fecha R2, SHA/esquema/permisos raw, estado/periodos oficiales, conservación del hallazgo IB09 y registro de CATALOG-TZ-001.
- Resultado independiente: **PASS_WITH_KNOWN_POST_FINDINGS**. El resultado describe la revisión de evidencia y no cambia el FAIL estructural de IB09.

## SHA-256 de artefactos POST

| Artefacto | SHA-256 |
|---|---|
| siar_station_audit.csv | afd25689ef92b1153cadde1b9ebaf11df1420058f2a79ac3432711f88af2d596 |
| siar_missing_days.csv | d3d94b39e030af7f7e5da8e7ed6a1eb047043de97f55f487a5d9715ab96b63fc |
| siar_dataset_manifest.csv | 52aae3ce411d91ac58b7fb71a3ec18e0ea24374b33b25146b4515a448262fb60 |

## Límite de cierre

**SIAR-QUALITY-CLOSEOUT = COMPLETE_BUT_INCOMPLETE_DATASET**. R2 y POST están terminados y certificados con hallazgos conocidos; Dataset Baleares v1 no se declara cerrado. Se mantiene el límite autorizado: no ejecutar una segunda pasada de huecos, no cambiar CSV raw, no modificar siar_worker.py ni generar otro snapshot POST sin nueva autorización.

