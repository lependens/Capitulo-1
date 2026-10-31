# Dónde consultar los datos y resultados

**Revisión de preservación: 2026-10-31.** Distinguir artefacto completo, evidencia compacta versionada y ubicación operativa externa. Esta guía no crea ni mueve datos.

| Qué consultar | Ubicación / evidencia | Alcance |
|---|---|---|
| Observaciones SIAR canónicas | [datos_siar_baleares/](../datos_siar_baleares/) | 12 CSV IB01–IB11 e IB101, 23 columnas; conservar originales |
| Catálogo reproducible | [estaciones_baleares.csv](../estaciones_baleares.csv) | Snapshot local; Info/ESTACIONES es autoridad a contrastar; Huso sigue con hallazgo abierto |
| Quality vigente | [POST corregido](../data_quality/siar/post/SIAR-QA-POST-20261020T193140Z/) | Tres CSV, paquete POST, informe y certificación; supersedes original sin borrarlo |
| POST original | [Snapshot superseded](../data_quality/siar/post/SIAR-QA-POST-20261018T233901Z/) | Historia preservada; no consumir su ledger como vigente |
| Dataset Observational v1 | [Índice y paths del productor](../scientific_pipeline/results/pm-vpd-20261023T194553Z/README.md) | JSON completo local y backup remoto SHA256 PASS en el scope aceptado; no está íntegro en Git |
| Predicciones de cuatro métodos | [Run completo e índice](../scientific_pipeline/results/four-classical-20261024T042250Z/artifact-index.json) | predictions.json y CSV masivos externos; Git no es copia completa del proyecto científico |
| Métricas y manifiesto de cuatro | [comparison/](../scientific_pipeline/results/four-classical-20261024T042250Z/comparison/) | metrics.json y manifest.json de bytes preservados; identidad y hashes en índice |
| Índice del run vigente de cuatro | [artifact-index.json](../scientific_pipeline/results/four-classical-20261024T042250Z/artifact-index.json) | Identidades/hashes de artefactos completos y compactos |
| Valiantzas y comparación clásica | [Resultados oficiales](../scientific_pipeline/results/classical-20261024T034529Z/README.md) | VALIANTZAS_V1, métricas/manifests en Git; CSV/JSON masivos externos con paths y hashes |
| Forecast SIAR raw | `/srv/siar-forecast/raw/` en servidor | Operativo append-only; no se presupone dentro de Git |
| Forecast AEMET raw | `/srv/aemet-forecast/raw/` en servidor | Wrapper/payload independientes; no se presupone dentro de Git |
| Backups | Drive cifrado (rclone crypt), copia adicional PC/OneDrive | Restore independiente demostrado; OneDrive no se etiqueta backup auditado independiente |

## Identidad del dataset publicado como evidencia

```text
dataset_id = siar-balears-observational-v1
dataset_sha256 = 0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb
quality_post = SIAR-QA-POST-20261020T193140Z
audit_source_repo_sha = 18e6d893d119d7a6bd2959d6fdeb69fbdfa2c865
source_repo_sha = 2c4874c73b6f818793cd2e0af42eeeb4cea9a6e9
original_pm_hs_pipeline_sha = 39c101e5999245eb15e3c6c1c867d90b4ac44468
four_method_pipeline_sha = a0b440fa3ab92e4593be521e4e819c6421327656
```

El productor registra el dataset completo bajo `scientific-results/dataset-v1-20261021T055924Z/dataset.json` y el run bajo `scientific-results/pm-vpd-20261023T194553Z/baselines/` dentro de su workspace Windows. Las rutas absolutas exactas están en el README del resultado enlazado arriba. Son localizaciones declaradas por 05 y respaldadas por manifiestos; esta misión documental no accede a ese disco ni certifica su disponibilidad actual.

El manifest original resuelve sus rutas relativas contra el **run completo**, no contra la copia compacta de Git. Conservar dataset, handoff, POST, catálogo, fuentes, parámetros y manifiestos para reproducir. El JSON contiene rutas absolutas de provenance: reconstruirlo en otro sitio puede cambiar su hash de bytes, aunque la procedencia y valores puedan verificarse; no reutilizar una identidad para bytes diferentes sin explicarlo.

## Convención propuesta — todavía no materializada

```text
scientific-results/
  observational-v1/
    dataset/
    baselines/
    exports/
```

Es una propuesta para acordar con el productor; no sustituye los paths reales anteriores ni autoriza mover outputs. Cada ejecución necesita identidad y directorio nuevos, sin sobrescritura. No crear una estructura vacía para hacer parecer que contiene artefactos.

## Rutas canónicas de preservación

[Arquitectura de backup](backup/architecture.md), [matriz de cobertura](backup/coverage.md) y [restore](backup/restore.md) son la referencia operativa detallada. Fuente: BACKUP-IMPLEMENTATION-INVENTORY-V1 PASS recibido, no inspección nueva.

| Función | Ubicación |
|---|---|
| Ciencia ingerida | `/srv/siar-project/scientific-artifacts` |
| Inbox | `/home/josep/project-artifact-inbox` |
| Receipts de ingestión | `/srv/siar-project/ingestion-receipts` |
| Manifest snapshot | `/srv/siar-project/preservation/snapshots/<backup_id>/manifest.json` |
| COMPLETE local | `/srv/siar-project/preservation/markers/<backup_id>/COMPLETE` |
| Receipt remoto | `/srv/siar-project/preservation/verified/<backup_id>.json` |
| Cuarentena restore | `/srv/siar-project/preservation/restores/<backup_id>_<suffix>/` |
| Remoto general lógico | `forecast-offsite-crypt:SIAR-project/snapshots/<backup_id>/` |
| Remoto AEMET específico | `forecast-offsite-crypt:AEMET-forecast/raw/runs` |

Windows automation DESIGN_READY / STILL_PENDING: `%LOCALAPPDATA%\SIAR-Science-Ingest\ready-for-backup` es raíz **propuesta**, no instalación. El inbox y el importador del servidor sí existen; no extender esa existencia al watcher Windows. [Diseño pendiente](backup/windows-ingestion.md).

## Recuperación y límites

GitHub es autoridad para código/docs/evidencia versionable; los raw forecast y archivos científicos masivos requieren preservación externa. Cada nuevo raw prospectivo relevante debe quedar cubierto por backup externo. Usar rclone copy, no sync, verificar hashes y restore, y conservar material crypt recuperable fuera del servidor.

STORAGE-DATA-PRESERVATION PASS: copia independiente fuera del Toshiba con 238/238 archivos de datos size/SHA256 PASS, 12/12 CSV contra POST y 242 archivos totales sin discrepancias. Copia descifrada preservada en PC personal. Esto no afirma que todo archivo nuevo futuro ya esté cubierto ni convierte OneDrive en un backup independientemente auditado.


## Exportación clásica — 24/10/2026

CSV completo con 75.496 filas preservado y verificado localmente:

```text
C:\Users\Josep\.codex\.chatgpt-projects\g-p-6ac0c6c1d8908191b03f324e51c278f1\scientific-results\classical-20261024T034529Z\comparison\classical-baselines.csv
```

SHA256 `66486b4138b7194ded3cd53f4d4d31f08e53ce3051d7c17ee43b30cdc957ed5f`.
[Índice de hashes y ubicaciones](../scientific_pipeline/results/classical-20261024T034529Z/artifact-index.json).
Pipeline comparación/VAL `4ac52e25c803cb2301316c5cb1353bf30cdb427a`; PM/HS conserva productor
`39c101e5999245eb15e3c6c1c867d90b4ac44468`. No es un nuevo dataset ni se han movido
artefactos anteriores. La verificación local no certifica backup externo nuevo.


## Exportación de cuatro métodos — 24/10/2026

[Manifests, métricas e índice](../scientific_pipeline/results/four-classical-20261024T042250Z/README.md).
CSV completo fuera de Git, 75.496 filas verificadas:

```text
C:\Users\Josep\.codex\.chatgpt-projects\g-p-6ac0c6c1d8908191b03f324e51c278f1\scientific-results\four-classical-20261024T042250Z\comparison\classical-baselines.csv
```

SHA256 `f55e4ed12691b84dbca4e090c49da538d5e7fad4f2ef371e1307a649cfb1af44`.
Pipeline HG-Rs/comparación `a0b440fa3ab92e4593be521e4e819c6421327656`.
Dataset v1 en su ubicación anterior, sin reconstrucción. El CSV de tres métodos
y sus hashes siguen preservados; este es un output nuevo, no una sobrescritura.

## PROJECT-ARTIFACT-BACKUP-V1 — PASS

Parte de cierre de 02 comunicado por Dirección 00: snapshot **backup_20261024T215639Z_63d0ef01**, **REMOTE_VERIFIED**. **128 archivos verificados = 82 scientific-results + 46 SIAR published Raw**. Scientific-results: **1,421,919,603 bytes**. El inventario del servidor tiene **221 entries**: no afirmar que los 221 estén respaldados.

Dataset v1 SHA256 `0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb` y CSV cuatro métodos SHA256 `f55e4ed12691b84dbca4e090c49da538d5e7fad4f2ef371e1307a649cfb1af44`: verificación PASS.

**cryptcheck PASS (128 archivos)** + restore independiente de **2 scientific artifacts y 2 operational references**, hashes PASS. Un intento redundante de verificación por descarga completa fue **INTERRUPTED**: **no declarar full-download verification PASS**. El snapshot se acepta por cryptcheck + restores independientes + SHA256 matches.

Preservación general para artefactos científicos **ya ingeridos al servidor**: **event-driven ingestion PASS, new=0, unchanged=9** en la prueba reportada; **daily preservation reconciliation ENABLED**. Es distinta del reconciliador SIAR Forecast de runs publicados cada **15 minutos**.

**LIMITACIÓN ABIERTA:** futuros resultados solo Windows = **NOT AUTOMATICALLY PROTECTED**. Requieren explicit transfer/ingestion → server preservation area → backup system → Drive crypt. No se garantiza todo resultado futuro ni se confunde inventario con cobertura.

Secret config y credentials **EXCLUDED** del snapshot. Material de recuperación crypt/OAuth **OFF-SERVER**; no publicar rclone.conf, refresh token, OAuth tokens ni crypt passwords. Closeout operativo: `/srv/siar-project/ingestion-receipts/closeout-e0417937.json`, referencia sin copiar contenido.

Toshiba ya no es la única copia **para los artefactos científicos y SIAR published Raw cubiertos por el snapshot verificado**. GitHub sigue siendo autoridad versionable, no copia íntegra de outputs masivos. Esta misión backup **no cambió Forecast**: timer captura y reconciliador ACTIVE; nuevo run automático real con backup/COMPLETE remoto aún pendiente. AEMET disabled/inactive, captures=0, gate=CLOSED, sin nuevos raw.


## Alcance real de backup operativo — parte 24/10

Reconciliador SIAR Forecast enabled/active cada 15 min, último Result success. rclone copy (no sync) sobre runs publicados pendientes en `/srv/siar-forecast/raw/runs/`, incluidos complete/partial/failed; excluye raw/tmp, CSV Sync, código general, /etc completo y resto del servidor.

OAuth Production, reauthorization PASS y CLI crypt listing PASS; copy/restore independiente SHA256 MATCH. Eso todavía no demuestra **nuevo run real automático → backup automático → COMPLETE remoto**; es el gate pendiente. El límite específico de siete días de OAuth Testing está eliminado, sin garantizar que tokens nunca puedan invalidarse. No versionar config/secretos.

AEMET: servicio específico `aemet-forecast-backup.service` static/inactive, sin timer independiente. BACKUP-IMPLEMENTATION-INVENTORY-V1 no observó hook de backup en el servicio de captura; queda corregida la afirmación anterior de backup tras cada intento. Runs AEMET incluidos también en preservación general, sin implicar nuevas capturas ni automatismo específico activo. Con capture timer disabled/inactive, nuevo backup automático AEMET NOT ACTIVE. Se habilitará reactivación tras SIAR-FORECAST-CAPTURE RUNNING. Estado y gates en [PROJECT_STATUS](PROJECT_STATUS.md).

## AHC temporal — 24/10/2026

[Publicación compacta, coeficientes, métricas y manifests](../scientific_pipeline/results/ahc-20261024T172400Z/README.md).
[Índice de todos los artefactos con bytes/ruta/SHA256](../scientific_pipeline/results/ahc-20261024T172400Z/artifact-index.json).
Pipeline AHC `38503dbe8422982deeb3680bbc3e82fd950da4e1`; Dataset v1
permanece en su ubicación anterior y con el mismo SHA256.

Raíz local de ejecución:

```text
C:\Users\Josep\.codex\.chatgpt-projects\g-p-6ac0c6c1d8908191b03f324e51c278f1\scientific-results\ahc-20261024T172400Z\
```

TRAIN/VALIDATION/TEST/cierre tienen directorios separados. `test/daily_predictions.csv`
conserva 75.496 filas (164.196.273 bytes). Métricas completas y desgloses
mensuales/estacionales/anuales permanecen fuera de Git; manifests y evidencia
compacta sí se versionan. Ese estado describe la publicación científica previa al cierre de preservación; posteriormente PROJECT-ARTIFACT-BACKUP-V1 PASS confirmó el scope remoto indicado arriba.
La cadena de manifests identifica fuentes, POST, dataset, protocolo y productores.

## AHC: ubicaciones y consumo vigente

[Resultado AHC](../scientific_pipeline/results/ahc-20261024T172400Z/README.md), [índice de artefactos](../scientific_pipeline/results/ahc-20261024T172400Z/artifact-index.json) y manifests de TRAIN/VALIDATION/TEST/closeout identifican los bytes científicos. Los paths del productor Windows siguen siendo válidos como localización de origen; ya no significan que Toshiba/Windows sea la única copia del scope aceptado.

TEST manifest SHA256: `502d7a39b82c8b5032931370ae8f05f4d8e7ecbbb93f6946810d0e47434fc469`. Closeout manifest SHA256: `39be9d706f962722862c50ce6b25fad5d0e93a57d954c4b71aa77cfe5942d6cd`. La evidencia compacta del run y sus referencias a backup pendiente se conservan como snapshot anterior al cierre remoto; no se alteran manifests ni índices para cambiar la historia.

El [contrato de visualización](../scientific_pipeline/docs/visualization_contract.md) está FROZEN/AVAILABLE TO 06: daily_predictions, métricas precalculadas, metadatos, cobertura, schemas y manifests. AHC_VALIDATED y AHC_CLIMATOLOGY tienen identidades separadas; el consumidor no recalcula ciencia ni usa climatología para atribuir métricas TEST.

## WEB-V0.1 — despliegue LAN 25/10/2026

- Aplicación: `http://192.168.1.43:8088/` (solo LAN).
- Checkout de producción: `/home/josep/siar-et0-web`, main `b5a73e3df273bb3922adbd31abf19ee9f65b8520`.
- Publicación derivada: `/home/josep/siar-et0-web-data/releases/web-v0.1-20261025`, montada como `/data:ro`.
- `observations.parquet`: SHA256 `6119e3521e8efb1251771a8d6fad2b5a5308e31d7eabeb66e0183c3add0deeca`.
- Fuente científica: CSV cuatro baselines SHA256 `f55e4ed12691b84dbca4e090c49da538d5e7fad4f2ef371e1307a649cfb1af44`; Dataset v1 SHA256 `0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb`.

El paquete WEB-V0.1 fue transferido explícitamente y verificado por hashes. Su presencia en el servidor no demuestra por sí sola backup remoto del nuevo derivado; debe entrar en el flujo de preservación antes de declarar esa copia protegida externamente.

Para nuevos artefactos: **explicit transfer/ingestion → server preservation area → backup system → Drive crypt**. Receipt operativo de cierre `/srv/siar-project/ingestion-receipts/closeout-e0417937.json`, sin copiar contenido sensible. La documentación recoge el parte de 02, no una inspección nueva del servidor.
