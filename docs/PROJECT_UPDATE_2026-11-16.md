# Actualización del proyecto — 16/11/2026

Estado comunicado por Dirección 00W a partir de los frentes 01, 02, 02B y 05. **REPORTED** significa comunicado, no inspeccionado por esta revisión; **VERIFIED** identifica únicamente la evidencia concreta que se indique; **PENDING** y **NO-GO/HOLD** mantienen gates abiertos. Los estados fechados anteriores se preservan como historia. No se inspeccionó el servidor, Kaggle ni los artefactos de ejecución en esta revisión.

## SIAR Sync V3 — desarrollo y preservación

V3 evolucionará desde v2.3.3, no desde cero. Se prevén autenticación con `.env` existente, worker/FastAPI/frontend, CSV de 23 columnas, catálogo oficial nacional, sincronización incremental y ausencias, ledger SQLite mínimo, publicación GitHub controlada, backup cifrado e interfaz de seguimiento. El histórico científico ET₀ V1/V2 permanece congelado. Se prevé preparar una copia inicial de CSV verificada en una raíz evolutiva distinta, reconocer fechas ya disponibles para evitar descargas duplicadas y añadir Baleares 2026 y después otras estaciones nacionales; **esta migración no consta ejecutada**.

**SIAR-V3-OFFSITE-PRESERVATION = PASS REPORTED** por 02B: backup `backup_20261114T191953Z_3677e506`, 19 componentes y 2 auxiliares, 21 archivos y 97.274 bytes, manifiesto SHA256 `7c3a2d04179463d0f2bdb9f64f417db5ac7024fa24394c6542002415a174723e`, destino `forecast-offsite-crypt:SIAR-project/snapshots/backup_20261114T191953Z_3677e506/`; restore independiente comunicado 21/21 MATCH. Fuente declarada `/home/josep/siar-v3-operational-package.K2htamLb/`. La preservación desbloquea desarrollo aislado, no acredita funcionamiento restaurado. **SIAR-SYNC-V3-DEVELOPMENT = IN_PROGRESS**, 52 tests PASS comunicados sobre lectura CSV, SHA256 y fechas; **SIAR-SYNC-RUNTIME = HOLD**. Las claves históricas son `REVOKED_OWNER_ATTESTED`, sin verificación técnica independiente afirmada.

## AEMET y SIAR Forecast

**AEMET A1 PASS REPORTED:** persistencia OAuth, renovación, backup cifrado y restore independiente comunicados por 02. **A2 NO PASS GLOBAL:** el run COMPLETE `run_20261115T091257.178143Z_69c30cfb63aa`, manifiesto SHA256 `3a58930149604903304b5b1241961852db065e5b34e8f4c4ff48556b7cb37270`, cuenta según 02wA con RAW publicado, validación municipal, backup cifrado, verificación remota y restore independiente PASS. El run anterior `run_20261114T194929.652746Z_65341b8e1082` permanece PARTIAL; el reconciliador informa `backup pending` y exit 1. Investigar, corregir y comprobar idempotencia antes del cierre A2.

**A3 GO condicionado**, no activado ni funcionando acreditado: tras A2 PASS, piloto exclusivo IB04 → Palma → 07040; tres capturas programadas verificadas antes de continuidad. Sin expansión territorial autorizada. Track B OPEN. Las [PR #72](https://github.com/lependens/Capitulo-1/pull/72), [#73](https://github.com/lependens/Capitulo-1/pull/73) y [#74](https://github.com/lependens/Capitulo-1/pull/74) figuraban OPEN / DRAFT / NOT MERGED en la consulta GitHub del 16/11; su existencia no prueba producción.

SIAR Forecast: el parte anterior de 02 comunicó PILOT RUNNING y capturas programadas con backups funcionales, y restore independiente sobre parte del inventario. No se afirma auditoría global de preservación PASS para runs o snapshots no verificados independientemente. No se ha inspeccionado el servicio en esta revisión.

## ET₀ V2 — DEVELOPMENT y gates

**ET0-V2-DEVELOPMENT-COMPUTATION = COMPLETE REPORTED** por 05; cuatro fases completadas computacionalmente en Kaggle `369953511`, código `1e419ad881db1cc9798685ef19f414862461674b`, Dataset v1 SHA256 `0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb`.

| Índice | Celdas | SHA256 comunicado |
|---|---:|---|
| AHC | 108 | `d475ec833ace324d8fb83ba038566c1e6a795707a12c8aa5272ebd485ef9c86e` |
| ML | 105 | `8a78db402354a9da42dc8be18237f02c5d12a813fdb448aa5da2688534fd4bd7` |
| ANN A | 105 | `0a10d6416c47f1745c4f717edbea429f8985b4211504ae82a96bc59a433f6f58` |
| ANN E | 105 | `81524c1cecb6fff1a97acba5f18168e943c2d4f6367d37fbb4f4485883f5f256` |

ANN E: 525/525 candidatos reportados. **ET0-V2-GLOBAL-FREEZE = LOCAL_VERIFIED_REPORTED**: `97acad7ce7cc39c65d82f6efdc36290ed0325caaef2dedb69d9ffbefe1addfbd`; manifest final `aac887af87a5dcd77fc85836e72fe7efef0ce4a613098a5cb2b8eb1a670eac42`; ZIP `f8997abb6d1e4d538ff86a2ea174d6636785dc7dd96cc4370ae545006f8e4b70`. 05 comunica verificación local de índices y `frozen.json` y publicación verificada en servidor; esta revisión no la reproduce.

**ET0-V2-FINAL-PRESERVATION = PENDING:** confirmar backup remoto y restauración independiente del ZIP; resolver o justificar ausencia de `__notebook__.ipynb` respecto al manifiesto; completar revisión científica independiente del freeze. **ET0-V2-RETROSPECTIVE = EXECUTED_VALIDATION_PENDING**: según parte posterior de 05 comunicado por Dirección el 16/11, Kaggle `370251881` completó la retrospectiva 2020–2025 con freeze SHA256 `97acad7ce7cc39c65d82f6efdc36290ed0325caaef2dedb69d9ffbefe1addfbd`, manifest SHA256 `f771dc3564270528c203d9af709b2b88649a552ef7dd831bf4503d04a2fd8daa`; predicciones, métricas e informe generados, sin reentrenamiento ni reselección con TEST según 05. Integridad completa, revisión científica, preservación externa y restauración siguen pendientes. Este parte no ha sido verificado aquí contra artefactos. No se declara resultado aprobado ni listo para la web. IB09 se trata como histórica separada, Quality V1 FAIL y verificación oficial independiente UNRESOLVED; IB07 e IB101 mantienen sus limitaciones temporales. AHC de histórico completo será in-sample. PM es referencia calculada, no ET₀ instrumental independiente; TEST es retrospectivo, no validación prospectiva independiente.

## Web y preservación

06 prepara rediseño web. Además de modelos, índices, manifiestos y checkpoints, 05 comunica predicciones, métricas e informe retrospectivos generados, aún pendientes de integridad, revisión científica, preservación y aprobación para publicación. La raíz prevista es `/srv/siar-project/scientific-artifacts/et0-v2/`; no se afirma creada. El contrato de publicación prevé separar `original/` inmutable, `derived/` con tablas CSV/Parquet, `catalog/` versionado con hashes y procedencia, y `evidence/`. AHC histórico completo se identifica como in-sample. La web no recalcula modelos ni usa resultados no aprobados; su integración depende de validación, preservación y publicación autorizada. WEB V0.3 es referencia actual de rollback. Preservación general 02B PASS/MAINTAINED y Windows automation PASS se limitan a su alcance demostrado y no prueban cobertura automática de cada resultado nuevo.
