# PROJECT STATUS — SIAR / ET₀

**Actualizado: 2026-10-24 (Europe/Madrid).** Fuente principal del estado vigente. Consolidación sobre main `9f3dde8e22ab8e8f76aeaa273b5e7681558ebbcf`; estado operativo según el último parte de Dirección 00 recibido en esta misión. Un PASS histórico no acredita que un timer siga ejecutándose hoy.

## Cómo leer el estado

- **VIGENTE:** evidencia o decisión aplicable actualmente.
- **HISTÓRICO:** snapshot fechado conservado para trazabilidad.
- **SUPERADO:** sustituido por una publicación o decisión posterior identificada.
- **PENDIENTE:** gate sin resultado demostrado.
- **LIMITACIÓN ABIERTA:** restricción conocida que no se debe resolver por inferencia.

[ROADMAP](ROADMAP.md) fija próximos gates; [MILESTONES](MILESTONES.md) conserva hitos append-only; [dónde consultar datos y resultados](DATA_LOCATIONS.md) identifica artefactos, ubicaciones y límites de recuperación. Las bitácoras `docs_1.* / docs_2.*` conservan etapas históricas y actualizaciones fechadas.

## Estado maestro por componente

| Área | Estado vigente | Próximo gate o límite |
|---|---|---|
| Quality SIAR Baleares | COMPLETE; POST corregido publicado mediante PR #29; 12 estaciones | CATALOG-TZ-001 OPEN / NON_BLOCKING; completitud observacional distinta de completitud de auditoría |
| Dataset Observational v1 | Construido y validado estructuralmente PASS; bytes preservados localmente | Conservar identidad, provenance y acceso a la copia completa; no se afirma que el JSON masivo esté en Git |
| Cuatro baselines v1 | PM_FAO56_V1, HS_RA_V1, VALIANTZAS_V1 y HG_RS_V1 oficiales PASS | Preservación de artefactos, AHC methodology review y contrato temporal antes de calibración |
| Preservación científica | Dataset/JSON/CSV completos en workspace Windows de 05; Git solo contiene evidencia compacta | PROJECT-ARTIFACT-BACKUP-V1 pendiente; no afirmar backup completo del proyecto |
| SIAR Sync | v2.3.3 canónica; SIAR-SYNC-RUNTIME = HOLD | SIAR-SYNC-PRODUCTION-ALIGNMENT + SIAR-SYNC-INCREMENTAL-2026 |
| SIAR Forecast | Timer de captura ACTIVE; reconciliador enabled/active cada 15 min, último Result success | Nuevo run real automático → backup automático → COMPLETE remoto; después CAPTURE RUNNING, mínimo dos ciclos PILOT RUNNING |
| AEMET Track A | E2E DEMONSTRATED; capture timer disabled/inactive; nuevo backup automático NOT ACTIVE | Reactivar tras SIAR-FORECAST-CAPTURE RUNNING; no depende de SIAR Sync |
| AEMET Track B | OPEN | Investigar feed AEMET exacto utilizado por SIAR, independientemente de Track A |
| Storage | Incidente CLOSED_OPERATIONALLY / MONITORING; preservación PASS; SSD IN SERVICE | Reabrir solo con nueva evidencia objetiva; sin plan de sustitución del SSD |
| ML / producto | Futuro; modelos históricos en legacy | Baselines clásicos comparables y split temporal sin leakage antes de nuevos modelos |

## Observaciones y Quality: cierre vigente

El snapshot científico 2025 está cerrado y no depende de reactivar la adquisición incremental 2026. Hay **12 CSV canónicos de 23 columnas**, IB01–IB11 e IB101, en `datos_siar_baleares/`; el catálogo reproducible es `estaciones_baleares.csv`. Info/ESTACIONES es la autoridad externa a contrastar.

Inventario: **9 ACTIVE** (IB01, IB02, IB03, IB04, IB05, IB06, IB08, IB10, IB11) y **3 CLOSED** (IB07, IB09, IB101). Cerradas: fin en Fecha_Baja; activas: cutoff explícito del snapshot.

POST vigente: **SIAR-QA-POST-20261020T193140Z**, que supersedes **SIAR-QA-POST-20261018T233901Z**. El original permanece preservado; su generador omitía del ledger la fecha RECOVERED que ya existía en raw. [PR #29](https://github.com/lependens/Capitulo-1/pull/29) publicó la corrección sin modificar raw.

| Identidad | Valor |
|---|---|
| Publicación canónica, source_repo_sha | `2c4874c73b6f818793cd2e0af42eeeb4cea9a6e9` |
| Snapshot auditado, audit_source_repo_sha | `18e6d893d119d7a6bd2959d6fdeb69fbdfa2c865` |
| Audit completeness | COMPLETE |
| Observational completeness | INCOMPLETE por ausencias SIAR confirmadas |
| Días esperados / observados oficiales | 77.840 / 75.495 |
| Filas raw preservadas | 75.496 |
| Ledger | 2.346 filas: 1 RECOVERED + 2.345 CONFIRMED_NO_DATA + 0 PENDING_RETRY |
| Cobertura | 96.987410 % |
| Estados de estación | 11 WARN, 1 FAIL (IB09), 0 INCOMPLETE |

**Una auditoría completa puede documentar una serie con menos del 100 % de observaciones.** Los 2.345 días sin dato están confirmados, no son consultas pendientes. PRE y R1/R2 son historia; Recovery no se presenta en ejecución.

IB07 / 2012-11-13 = **RECOVERED**, evidencia FORENSIC-IB07-001/F2, comparación canónica 23/23 campos. La confirmación del contenido por consulta independiente no demuestra la procedencia de la escritura original, que sigue registrada como no resuelta en su evidencia.

IB09 / 2005-10-23 permanece raw PRESERVED y OUTSIDE_OFFICIAL_PERIOD; su instalación oficial es 2005-10-24 y baja 2018-10-10. El FAIL de estación se conserva. El dataset no borra esa fila; la política actual de evaluación excluye todas las filas IB09 por Quality FAIL, limitación explícita del run oficial.

**CATALOG-TZ-001 = OPEN / NON_BLOCKING:** snapshot local Huso 31 frente a Info/ESTACIONES 30. No autocorregir catálogo ni usar ese campo científicamente sin resolver su significado. No bloquea por sí solo el cierre.

El saneamiento operativo posterior a v2.3.2 está completado: 12/12 CSV 0644; hashes de contenido PRE = POST 12/12 durante chmod. No es una versión funcional ni una afirmación de que Recovery nunca añadiera datos después. Ver [estado observacional](siar/sync-status.md), [contrato Quality](siar/quality.md) y [POST corregido](../data_quality/siar/post/SIAR-QA-POST-20261020T193140Z/final_consolidated_report.md).

## Dataset y resultados científicos

Dataset **siar-balears-observational-v1**, construido con handoff corregido y validación estructural PASS:

```text
SHA256 = 0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb
```

Preserva 75.496 filas observacionales, 75.495 dentro de periodo y 1 exterior, más el ledger de 2.346 registros. Conserva las 23 columnas originales y añade estados, Quality, provenance y elegibilidad. No interpola, imputa ni fabrica meteorología para días ausentes.

### Cuatro baselines oficiales

PR #30/#31: tolerancia numérica VPD y primera ejecución PM/HS; PR #33/#34: VALIANTZAS_V1 y comparación de tres; PR #35/#36: HG_RS_V1 y publicación de cuatro métodos. Dataset, raw y Quality permanecen intactos; no hay calibración local, AHC ni ML.

Pipeline HG-Rs/comparación: `a0b440fa3ab92e4593be521e4e819c6421327656`; publicación de cuatro métodos: `9f3dde8e22ab8e8f76aeaa273b5e7681558ebbcf`. **75 tests PASS; CI Python 3.11/3.12/3.13 PASS** según [evidencia oficial](../scientific_pipeline/results/four-classical-20261024T042250Z/README.md).

Comparación principal **PM_HS_VAL_HGRS, n=70.860**, misma cohorte para los cuatro. MAE/RMSE/bias en mm/día; error=predicción−EtPMon; R²=1−SSE/SST.

| Método | n común | MAE | RMSE | bias / MBE | R² predictivo |
|---|---:|---:|---:|---:|---:|
| PM_FAO56_V1 | 70860 | 0.011540 | 0.028298 | -0.001565 | 0.999735 |
| HS_RA_V1 | 70860 | 0.712843 | 0.874359 | +0.461627 | 0.746559 |
| VALIANTZAS_V1 | 70860 | 0.455952 | 0.608984 | +0.168919 | 0.877055 |
| HG_RS_V1 | 70860 | 0.471171 | 0.565637 | +0.275804 | 0.893935 |

HG_RS_V1: 70.978 predicciones, 70.912 evaluables propios y 0 fallos. No confundir esa cohorte propia con la común de cuatro.

HG-Rs obtiene mejor RMSE/R² que Valiantzas; Valiantzas obtiene mejor MAE y menor bias absoluto en esta cohorte. **No hay vencedor universal.** EtPMon es referencia calculada SIAR, no ET₀ instrumental independiente; la coincidencia PM↔EtPMon no es validación independiente.

| Método | Información principal |
|---|---|
| PM_FAO56_V1 | Temperaturas, Rs medida, humedad, viento, presión/altitud y geometría |
| HS_RA_V1 | Temperaturas y Ra geométrica; sin Rs medida ni humedad |
| VALIANTZAS_V1 | Temperaturas, Ra y HumedadMedia; estima radiación empíricamente |
| HG_RS_V1 | Rs medida y Tcalc; sin Ra, sqrt(TD), humedad o viento |

Los cuatro usan Tcalc=(Tmax+Tmin)/2. HS conserva 0.0023×0.408×Ra×(Tcalc+17.8)×sqrt(Tmax−Tmin); HG-Rs usa 0.0135×0.408×Rs×(Tcalc+17.8). VAL v1 adopta Martí et al. (2024), Ecs.1b+3, factor energético 0.408, coeficiente expandido 0.001861704 y (1.001−RHmedia/100)^0.2; **la fórmula legacy no es su autoridad**. [Contrato VAL](../scientific_pipeline/docs/valiantzas_v1.md) y [HG-Rs](../scientific_pipeline/docs/hargreaves_rs_v1.md).

Una mayor similitud de inputs con la referencia puede favorecer concordancia; es interpretación, no causalidad demostrada ni «más inputs siempre mejora». Separar accuracy against EtPMon de input requirements / operational availability. Sigue interesando qué precisión se consigue con menos variables.

Permanecen UNITS_PROVENANCE_PARTIAL, viento 2 m por convención sin verificación instrumental exhaustiva, IB09 excluida por Quality FAIL, cohortes desiguales y periodo corto IB101. No se resuelven esas limitaciones al añadir métodos.

### AHC: revisión separada, sin nueva metodología decidida

El AHC legacy era mean(ET0_calc/ET0_empirical) por estación sobre toda la serie; después multiplicaba las estimaciones y evaluaba sobre los mismos días. Es **HISTÓRICO / full-series / in-sample**, no validación predictiva.

**AHC-METHODOLOGY-REVIEW pendiente:** bibliografía, coeficiente climatológico de estación frente a calibración predictiva, estabilidad temporal y opciones anual/estacional/mensual si procede. Si se estima de datos, AHC es un parámetro calibrado. La misión científica específica definirá estimación, datos de ajuste y validación temporal antes de AHC_V1; esta PR no decide todo el dataset frente a solo TRAIN.

## SIAR Sync: HOLD independiente del snapshot científico

SIAR Sync v2.3.3 mantiene autenticación dinámica, bloques ≤7 días, esquema 23 columnas, escritura atómica, reintentos HTTP, persistencia local ante fallos Git y checkpoints. SIAR_CA_BUNDLE es opcional y la verificación TLS permanece activa; no se usa verify=False. INC-INFRA-001 está resuelta e init: true es canónico.

**SIAR-SYNC-RUNTIME = HOLD.** Diagnóstico de 02 comunicado por Dirección 00: checkout residual `quality/recovery-20261012T085131Z`, HEAD `18e6d893...`; main ya contiene la recuperación publicada. El worker tiene target default 2025-12-31 y no hay modo incremental 2026 demostrado que vaya de último dato útil a actualidad sin reabrir huecos históricos. /start podría intentar aproximadamente 1.958 huecos históricos de activas.

El panel mezcla historical gaps con current/prospective lag y no aplica correctamente Fecha_Baja a cerradas, aunque el worker sí. Reactivar exige **SIAR-SYNC-PRODUCTION-ALIGNMENT + SIAR-SYNC-INCREMENTAL-2026**; no ejecutar /start como actualización rutinaria. Este documento no despliega ni actualiza el checkout del servidor.

## Forecast SIAR: adquisición prospectiva desacoplada

Collector independiente en `forecast_collector/`, runtime permanente `/opt/siar-forecast`, configuración `/etc/siar-forecast`, raw `/srv/siar-forecast/raw/`, identidad dedicada siar-forecast. SMOKE-FC-001, BACKUP-FC-001 y FC-STATIONS-001 PASS. Append-only, hashes, schemas, TLS, publicación atómica y restore independiente demostrados.

Ocho estaciones funcionales: IB01, IB02, IB03, IB04, IB05, IB06, IB10, IB11. IB08 sigue **ACTIVE / TEMPORARILY_UNAVAILABLE_EXTERNAL** por indisponibilidad externa observada en su combinación de catálogo; no bloquea las demás ni se transforma en CLOSED. No se conoce la causa interna exacta de SIAR/MAPA.

**Parte operativo vigente de Dirección 00 — 24/10:** timer SIAR Forecast **ACTIVE**. No se sigue presentando como pausado. Esto no demuestra por sí solo **SIAR-FORECAST-CAPTURE = RUNNING**.

El reconciliador de backup está **enabled / active**, cada **15 minutos**, último Result observado **success**. Usa **rclone copy, nunca sync**, y procesa runs publicados pendientes bajo `/srv/siar-forecast/raw/runs/`, incluidos complete, partial y failed. Excluye raw/tmp, CSV SIAR Sync, código general, /etc completo y resto del servidor. No es un backup global del proyecto.

OAuth Google/rclone: application status **Production**, reauthorization **PASS**, CLI crypt listing **PASS**. El límite específico de siete días procedía de Testing y queda eliminado tras Production + nueva autorización; no significa que tokens nunca puedan invalidarse. Config reportada por 02, SHA256 `1c2f095f37082deaa94b3fcaeb3dfeaf19e917f3b5e6419d6725aafe9839b45c`; no publicar config ni secretos.

Prueba runtime: copy to crypt PASS → independent restore PASS → SHA256 MATCH. **Falta demostrar un nuevo run Forecast SIAR real automático → backup automático → COMPLETE remoto.**

- Ese primer ciclo completo PASS habilita **SIAR-FORECAST-CAPTURE = RUNNING**.
- Al menos dos ciclos automáticos PASS habilitan **SIAR-FORECAST-PILOT = RUNNING**.

Forecast es independiente de SIAR Sync HOLD. Cadencia experimental de captura 08:00/14:00/20:00 Europe/Madrid, distinta de la cadencia de reconciliación de 15 min. No reconstruir emisiones perdidas; source issue timestamp puede permanecer null y D+0…D+6 sigue sin asignarse. HTML persisted/omitted_sensitive/not_available según contrato.

## AEMET: Track A y Track B

Track A E2E DEMONSTRATED; PR #18 Collector, #19 comparación id municipal, #20 permisos POSIX. Wrapper y payload raw append-only con bytes, timestamps y provenance preservados. IB04 Son Ferriol → Palma → municipio 07040 representa la capital municipal, no la estación exacta.

Último parte 24/10: **capture timer disabled / inactive**. Backup AEMET se ejecuta después de cada intento de captura, success o failure; **no hay timer de backup independiente**. Por estar deshabilitada la captura, **new AEMET automatic backup = NOT ACTIVE**. La dependencia vigente es **SIAR-FORECAST-CAPTURE = RUNNING → reactivar AEMET Track A prospectivo**. No necesita SIAR Sync ni esperar a SIAR-FORECAST-PILOT RUNNING. Se conserva raw independiente en `/srv/aemet-forecast/raw/` y cada nueva captura relevante debe quedar cubierta por backup externo.

No están demostrados ET₀, D+n, timezone científico de emisión, recuperación histórica ni entradas suficientes para PM completo. No se infiere timezone ni se integra científicamente con SIAR. **Track B = OPEN**, investigación del feed exacto usado por SIAR, independiente de Track A.

## Storage y preservación

Incidente de storage: **CLOSED_OPERATIONALLY / MONITORING**; **SSD = IN SERVICE**. El parte recibido no aporta evidencia actual de fallo físico activo del Toshiba. SMART PASS; sin sectores reasignados, Reported_Uncorrect, CRC errors ni error log; raíz rw y sin nuevos I/O/ext4 errors observados.

**STORAGE-DATA-PRESERVATION = PASS:** copia independiente fuera del Toshiba, 238/238 archivos de datos size + SHA256 PASS, 12/12 CSV contra POST PASS y 242 archivos totales sin ausencias/discrepancias. Copia descifrada preservada en PC personal. No convierte el incidente cerrado en una línea activa del roadmap.

STOP de infraestructura si reaparecen I/O error, emergency_ro, SMART warning, USB reset asociado o corrupción. Monitorización no equivale a prometer ausencia de fallos futuros.

GitHub conserva código/docs/artefactos versionables. Google Drive con rclone crypt tiene copias verificadas de raw/preservación dentro de sus alcances demostrados; rclone copy, no sync; contenido/nombres/directorios cifrados y restore independiente demostrado. PC personal + OneDrive es copia adicional: no se presenta OneDrive como backup auditado independiente sin verificación expresa. La recuperabilidad externa del material crypt y la cobertura de nuevos raw mantienen validaciones propias.

## Próximos gates y gobierno

1. **PROJECT-ARTIFACT-BACKUP-V1:** prioridad de preservación del dataset completo, handoff científico, scientific-results completos, CSV exports, predictions, manifests, futuros ML artifacts, raw Forecast y configuración operativa recuperable. 02 recibirá misión separada. **GitHub NO es una copia completa del proyecto científico y no se afirma backup de todos esos artefactos.**
2. **AHC-METHODOLOGY-REVIEW**, después contrato de evaluación temporal, modelos calibrados simples, regresión/árboles/boosting y ANN más adelante. No saltar directamente de baselines a ANN.
3. 02: demostrar nuevo ciclo SIAR real con backup automático/COMPLETE remoto; tras CAPTURE RUNNING reactivar AEMET. Dos ciclos para PILOT RUNNING.
4. Resolver SIAR-SYNC-PRODUCTION-ALIGNMENT + SIAR-SYNC-INCREMENTAL-2026 antes de levantar HOLD, sin reabrir Recovery.
5. Acumular forecast prospectivo → issue/valid time/D+n → datasets derivados → comparación con observaciones posteriores.
6. Producto Web/API, ET₀+Kc, ETc y riego tras validación.

Los workstreams trabajan autónomamente dentro de objetivos/invariantes/STOP. Escalar arquitectura, seguridad, integridad raw y semántica científica. Gates requieren evidencia: código/timer activo no equivalen a run/backups demostrados. Las ubicaciones y límites de preservación están en [DATA_LOCATIONS](DATA_LOCATIONS.md).

## Registros de ampliación — trazabilidad

### Comparación clásica oficial — 24/10/2026

[Resultado, cohortes, métricas, CSV y hashes](../scientific_pipeline/results/classical-20261024T034529Z/README.md).
Dataset v1 intacto; PM/HS oficiales reutilizados; VAL 70.944 predicciones,
70.878 evaluables, 0 fallos. Martí Ecs.1b+3 con 0.408, sin calibración.
Pipeline 4ac52e25c803cb2301316c5cb1353bf30cdb427a; ambas identidades audit/canonical preservadas.


### Ampliación HG-Rs — 24/10/2026

[Comparación de cuatro modelos y hashes](../scientific_pipeline/results/four-classical-20261024T042250Z/README.md).
HG_RS_V1 70978 calculadas, 70912 evaluables, 0 fallos;
cohorte común 70860. PR #35, pipeline `a0b440fa3ab92e4593be521e4e819c6421327656`.
Dataset y todos los resultados PM/HS/VAL anteriores intactos, sin calibración.
75 tests/CI PASS. Ambos SHA audit/canonical preservados.
