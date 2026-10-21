# Quality Audit v1 POST

- Audit ID: SIAR-QA-POST-20261020T193140Z
- Audit UTC: 2026-10-20T19:40:51Z
- Repository SHA: 18e6d893d119d7a6bd2959d6fdeb69fbdfa2c865
- PRE audit/workspace: SIAR-QA-PRE-20261012T064810Z; SHA e850827e24cff11ae46f51c9bc3e8a2f36b0d2da
- PRE gaps: 2346
- Raw inventory: raw_final_inventory.csv
- POST result: **FAIL**

## Station results

| Station | State | Expected start | Expected end | Raw rows | Observed days | Missing days | Coverage % | Audit | Outside-period rows |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| IB01 | ACTIVE | 2004-03-30 | 2025-12-31 | 7716 | 7716 | 231 | 97.09 | WARN | 0 |
| IB02 | ACTIVE | 2004-03-31 | 2025-12-31 | 7697 | 7697 | 249 | 96.87 | WARN | 0 |
| IB03 | ACTIVE | 2004-04-01 | 2025-12-31 | 7873 | 7873 | 72 | 99.09 | WARN | 0 |
| IB04 | ACTIVE | 2004-04-01 | 2025-12-31 | 7729 | 7729 | 216 | 97.28 | WARN | 0 |
| IB05 | ACTIVE | 2004-04-05 | 2025-12-31 | 7834 | 7834 | 107 | 98.65 | WARN | 0 |
| IB06 | ACTIVE | 2004-04-05 | 2025-12-31 | 7862 | 7862 | 79 | 99.01 | WARN | 0 |
| IB07 | CLOSED | 2004-08-11 | 2012-11-13 | 2946 | 2946 | 71 | 97.65 | WARN | 0 |
| IB08 | ACTIVE | 2005-10-05 | 2025-12-31 | 7123 | 7123 | 270 | 96.35 | WARN | 0 |
| IB09 | CLOSED | 2005-10-24 | 2018-10-10 | 4423 | 4422 | 313 | 93.39 | FAIL | 1 |
| IB10 | ACTIVE | 2005-10-08 | 2025-12-31 | 7007 | 7007 | 383 | 94.82 | WARN | 0 |
| IB101 | CLOSED | 2008-12-02 | 2009-08-10 | 249 | 249 | 3 | 98.81 | WARN | 0 |
| IB11 | ACTIVE | 2005-10-10 | 2025-12-31 | 7037 | 7037 | 351 | 95.25 | WARN | 0 |

## Global totals

- Raw rows: 75496
- Official station-days: 77840
- Observed station-days: 75495
- Coverage: 96.98741%
- Unobserved dates (PENDING_RETRY + CONFIRMED_NO_DATA): 2345; gap ledger rows including RECOVERED: 2346
- Gap states: {"RECOVERED": 1, "PENDING_RETRY": 0, "CONFIRMED_NO_DATA": 2345}
- Station audit statuses: {"FAIL": 1, "INCOMPLETE": 0, "WARN": 11, "PASS": 0}
- Outside official period: 1 (IB09 anchor 2005-10-23 preserved)

## Recovery trace

- R1/R2 query count: 2782
- API result counts: {"R1/HTTP_ERROR": 2, "R1/ZERO_ROWS": 429, "R2/ZERO_ROWS": 2345, "R2/INVALID_RESPONSE": 5, "R2/HTTP_ERROR": 1}
- Info/ACCESOS snapshots: 10
- Latest quota snapshot: {"audit_id": "SIAR-QA-POST-20261012T085131Z", "phase": "R2", "query_id": "QUOTA-INITIAL", "station_code": "", "range": "", "expected_records": 1, "query": "Info/ACCESOS", "started_at_utc": "2026-10-18T20:50:16.565875Z", "completed_at_utc": "2026-10-18T20:50:17.642704Z", "http_requests": [{"timestamp_utc": "2026-10-18T20:50:16.565951Z", "path": "/Info/ACCESOS", "http_status": 200, "exception": ""}], "api_result": "OK", "quota_raw": {"MaxAccesosMinuto": 5, "NumAccesosMinutoActual": 1, "MaxAccesosDia": 7200, "NumAccesosDiaActual": 1010, "MaxRegistrosMinuto": 100, "RegistrosAcumuladosMinuto": 0, "MaxRegistrosDia": 10000, "RegistrosAcumuladosDia": 606}, "quota": {"peticiones_minuto": 1, "max_peticiones_minuto": 5, "peticiones_dia": 1010, "max_peticiones_dia": 7200, "registros_minuto": 0, "max_registros_minuto": 100, "registros_dia": 606, "max_registros_dia": 10000}, "error": null}
- Per-date trace: recovery_trace.csv
- Execution events: recovery_execution_events.jsonl
- Git checkpoints: see trace git_commit column and repository history.

## Permissions and immutability

- Raw file modes after recovery: {"0644": 12}
- PRE artifact SHA-256 values: {"siar_station_audit.csv": "ffa01ca0313bf7416729caf36cd0737cd2221bb71a75bb28d4fa6d58a5136efa", "siar_missing_days.csv": "e172c213725dde7425470bb08321004690a25580efa55122219147a88888edd7", "siar_dataset_manifest.csv": "de57553e02bece7bd9f0bbd4688451639fe16fdb60b8b51e5e855f2351a72169"}
- PRE artifacts were hash-checked before POST generation; no PRE output was overwritten.
- Canonical schema: 23 columns; raw hashes and row counts are in raw_final_inventory.csv.
- No station catalog, date conversion, period, or worker edits were made.

## CATALOG-TZ-001

- Passive finding: local catalog Huso=31; Info/ESTACIONES Huso=30; 12/12 stations affected.
- No timezone conversion or catalog change was performed.
- Evidence: official_evidence/catalog_tz_001_evidence.csv and CATALOG-TZ-001.md.

## Recovery boundary

- Only the 2,346 PRE PENDING_RETRY dates were planned for R1/R2.
- R1 range responses with zero records did not change gap states to CONFIRMED_NO_DATA.
- R2 confirms no-data only from valid single-date zero-row responses; failed/incomplete attempts remain pending.
- RECOVERED means the requested date was verified present in the canonical raw CSV after merge.
- No interpolation or synthetic zeroes; existing rows are checked for preservation.

## Gate

**STOP after Quality Audit POST.** No second gap pass, no further recovery, no raw edits, no POST snapshot, and no Dataset Baleares v1 closure.
