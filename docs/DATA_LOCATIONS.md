# Dónde consultar los datos y resultados

**Vigente: 2026-10-24.** Distinguir artefacto completo, evidencia compacta versionada y ubicación operativa externa. Esta guía no crea ni mueve datos.

| Qué consultar | Ubicación / evidencia | Alcance |
|---|---|---|
| Observaciones SIAR canónicas | [datos_siar_baleares/](../datos_siar_baleares/) | 12 CSV IB01–IB11 e IB101, 23 columnas; conservar originales |
| Catálogo reproducible | [estaciones_baleares.csv](../estaciones_baleares.csv) | Snapshot local; Info/ESTACIONES es autoridad a contrastar; Huso sigue con hallazgo abierto |
| Quality vigente | [POST corregido](../data_quality/siar/post/SIAR-QA-POST-20261020T193140Z/) | Tres CSV, paquete POST, informe y certificación; supersedes original sin borrarlo |
| POST original | [Snapshot superseded](../data_quality/siar/post/SIAR-QA-POST-20261018T233901Z/) | Historia preservada; no consumir su ledger como vigente |
| Dataset Observational v1 | [Índice y paths del productor](../scientific_pipeline/results/pm-vpd-20261023T194553Z/README.md) | JSON observational-v1 completo preservado localmente; no está íntegro versionado en Git |
| Predicciones PM/HS | Run completo externo indicado por el productor | predictions.json masivo no incluido en evidencia Git compacta |
| Métricas y manifiesto | [baselines/](../scientific_pipeline/results/pm-vpd-20261023T194553Z/baselines/) | metrics.json y manifest.json de bytes preservados; identidad y hashes en índice |
| Índice del run | [artifact-index.json](../scientific_pipeline/results/pm-vpd-20261023T194553Z/artifact-index.json) | Identidades/hashes de artefactos completos y compactos |
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
pipeline_sha = 39c101e5999245eb15e3c6c1c867d90b4ac44468
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
