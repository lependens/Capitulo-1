# Informes de calidad SIAR

## Estado vigente — 23/10/2026

Quality completeness COMPLETE; serie observacional INCOMPLETE por ausencias SIAR confirmadas. POST vigente [SIAR-QA-POST-20261020T193140Z](post/SIAR-QA-POST-20261020T193140Z/): 75.496 filas raw, 75.495 oficiales, cobertura 96.987410 %, ledger 2.346 = 1 RECOVERED + 2.345 CONFIRMED_NO_DATA + 0 PENDING_RETRY. IB07/2012-11-13 F2 verificado 23/23; IB09/2005-10-23 preservado fuera de periodo; CATALOG-TZ-001 OPEN / NON_BLOCKING.

Publicación canónica `2c4874c73b6f818793cd2e0af42eeeb4cea9a6e9`; SHA auditado `18e6d893d119d7a6bd2959d6fdeb69fbdfa2c865`. Dataset v1 ya construido/validado y run PM/HS oficial PASS: [datos y resultados](../../docs/DATA_LOCATIONS.md).

**Historia preservada:** el resumen original comunicado abajo incluía RECOVERED=1, pero su CSV ledger omitía esa fila; no utilizar esos hashes originales como contrato vigente. La corrección superseding es la autoridad. Ningún artefacto POST se modifica en esta actualización.

## Corrección superseding del ledger

El POST original SIAR-QA-POST-20261018T233901Z permanece preservado byte a byte. La versión vigente para el ledger es [SIAR-QA-POST-20261020T193140Z](post/SIAR-QA-POST-20261020T193140Z/), que lo supersedes y restaura la fila persistida IB07 / 2012-11-13 = RECOVERED. La causa y el fix reproducible están en [ledger-correction.md](ledger-correction.md). Los estados y la semántica de auditoría no cambiaron.

El POST original, ahora superseded, permanece preservado byte a byte en [post/SIAR-QA-POST-20261018T233901Z/](post/SIAR-QA-POST-20261018T233901Z/):

- siar_station_audit.csv
- siar_missing_days.csv
- siar_dataset_manifest.csv
- final_consolidated_report.md
- quality-post-package.json: índice de los tres informes que usa el validador scientific_pipeline.

El audit_id es SIAR-QA-POST-20261018T233901Z; su audit_source_repo_sha es 18e6d893d119d7a6bd2959d6fdeb69fbdfa2c865.

## Resultado del POST original (superseded)

- Audit completeness: COMPLETE
- Observational completeness: 96.987410 % (75.495 de 77.840 días)
- RECOVERED: 1
- CONFIRMED_NO_DATA: 2.345
- PENDING_RETRY: 0
- Unresolved evidence: 0
- Estado consolidado: COMPLETE_BUT_INCOMPLETE_DATASET

COMPLETE_BUT_INCOMPLETE_DATASET significa que la auditoría terminó y que la serie tiene cobertura inferior al 100 % debido a días confirmados sin registros SIAR; no significa que la auditoría esté incompleta. CONFIRMED_NO_DATA es un estado terminal válido.

Los SHA-256 aceptados son:

| Artefacto | SHA-256 |
|---|---|
| siar_station_audit.csv | afd25689ef92b1153cadde1b9ebaf11df1420058f2a79ac3432711f88af2d596 |
| siar_missing_days.csv | d3d94b39e030af7f7e5da8e7ed6a1eb047043de97f55f487a5d9715ab96b63fc |
| siar_dataset_manifest.csv | 52aae3ce411d91ac58b7fb71a3ec18e0ea24374b33b25146b4515a448262fb60 |

final_consolidated_report.md conserva SHA-256 6c64d0bc86174f6223942c05f07106136689b417544be169339ab108e899e003.

## Hallazgos

- IB09 / 2005-10-23: raw preservado; clasificación científica OUTSIDE_OFFICIAL_PERIOD. Queda fuera del periodo oficial.
- CATALOG-TZ-001: OPEN / NON_BLOCKING. El Huso del catálogo local es 31 e Info/ESTACIONES informa 30 para 12/12 estaciones; no se autocorrige el catálogo.

El CSV completo de IB07 publicado con este snapshot coincide con el SHA canónico del manifest. La observación IB07 / 2012-11-13 está clasificada como RECOVERED.

Las columnas y reglas de auditoría están definidas en [Quality Audit v1](../../docs/siar/quality.md). El estado operativo está en [sync-status](../../docs/siar/sync-status.md). Esta publicación no genera Dataset Observational v1.
