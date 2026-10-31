# PROJECT STATUS — SIAR / ET₀

**Actualizado: 2026-10-31 (Europe/Madrid), consolidación hasta PR #49.** Base releída: `2da62f17314e77461d2fe9cc405127e1f5870fdc`. Estado operativo y hashes externos según el parte vigente de Dirección 00; esta actualización no inspecciona ni cambia producción. Un runtime fusionado no demuestra una campaña científica completada; un PASS histórico no acredita que un timer siga ejecutándose hoy.

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
| Dataset Observational v1 | Construido/validado PASS; SHA256 verificado en snapshot remoto aceptado | Conservar identidad, provenance y acceso a la copia completa; no se afirma que el JSON masivo esté en Git |
| Ciencia ET₀ | Dataset v1, cuatro baselines y AHC temporal PASS; rolling validation, Linux smoke y runtime ML PASS | Campaña ML clásica oficial PENDING EXECUTION, pendiente de despliegue/ejecución del runtime; ANN NOT STARTED |
| Preservación científica | PROJECT-ARTIFACT-BACKUP-V1 PASS; 128 archivos REMOTE_VERIFIED | Nuevos resultados solo Windows requieren transferencia/ingestión explícita |
| SIAR Sync | v2.3.3 canónica; SIAR-SYNC-RUNTIME = HOLD | SIAR-SYNC-PRODUCTION-ALIGNMENT + SIAR-SYNC-INCREMENTAL-2026 |
| SIAR Forecast | Timer de captura ACTIVE; reconciliador enabled/active cada 15 min, último Result success | Nuevo run real automático → backup automático → COMPLETE remoto; después CAPTURE RUNNING, mínimo dos ciclos PILOT RUNNING |
| AEMET Track A | E2E DEMONSTRATED; capture timer disabled/inactive; nuevo backup automático NOT ACTIVE | Reactivar tras SIAR-FORECAST-CAPTURE RUNNING; no depende de SIAR Sync |
| AEMET Track B | OPEN | Investigar feed AEMET exacto utilizado por SIAR, independientemente de Track A |
| Storage | Incidente CLOSED_OPERATIONALLY / MONITORING; preservación PASS; SSD IN SERVICE | Reabrir solo con nueva evidencia objetiva; sin plan de sustitución del SSD |
| Producto | WEB-V0.2-AHC PASS / DEPLOYED; cuatro baselines y AHC publicados, LAN only, datos read-only | WEB-V0.3-ML espera outputs ML oficiales publicados; integración Forecast posterior |

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

### AHC — metodología y validación temporal PASS

**AHC-METHODOLOGY-AND-TEMPORAL-VALIDATION = PASS**, PR #38/#39, pipeline `38503dbe8422982deeb3680bbc3e82fd950da4e1`, publicación `d15bbffac99164994aa884c1a47dc680d54e66af`. **86 tests y CI Python 3.11/3.12/3.13 PASS**. [Resultado oficial](../scientific_pipeline/results/ahc-20261024T172400Z/README.md). Raw, Quality, Dataset v1 y cuatro baselines intactos.

AHC_VALIDATED: **a=Σ(x·PM)/Σ(x²)**, OLS sin intercepto por estación; **ET0_adjusted=a×x**. Referencia primaria PM_FAO56_V1; EtPMon secundaria calculada SIAR, ninguna ET medida independiente. Legacy mean(PM/x), full-series/in-sample, queda HISTÓRICO.

Cortes globales cronológicos **TRAIN 2004–2015 / VALIDATION 2016–2020 / TEST 2021–2025**. Ajuste solo TRAIN, congelado antes de VALIDATION/TEST; sin random split ni refit TRAIN+VALIDATION. TEST **retrospective_locked**, no test prospectivo nunca visto porque el histórico ya fue inspeccionado.

TEST común contra PM, **n=15.958**, ajustes anuales; MAE/RMSE/bias en mm/día:

| Método | MAE | RMSE | bias | R² |
|---|---:|---:|---:|---:|
| HS_RA_AHC_VALIDATED | 0.504081 | 0.638838 | 0.230566 | 0.860481 |
| HG_RS_AHC_VALIDATED | 0.318930 | 0.414241 | 0.003436 | 0.941338 |
| VALIANTZAS_AHC_VALIDATED | 0.392259 | 0.523955 | 0.166835 | 0.906149 |

VALIDATION seleccionó **HS-Ra annual / HG-Rs seasonal / Valiantzas annual**. HG_RS_AHC_VALIDATED_SELECTED estacional: TEST n=15.958, MAE 0.306919, RMSE 0.401392, bias −0.011504, R² 0.944921. Mensual no justificó complejidad adicional; no reseleccionar después de TEST ni sustituir el benchmark anual.

Estabilidad **PASS**, CV muestral máximo **6.612 %**, max/min **1.164415**. No equivale a estabilidad del error: se conservan empeoramientos, **IB11/2025/HS RMSE 0.756→0.897 (≈+18.57 %)**.

**AHC_CLIMATOLOGY** solo se calculó después de completar TEST usando todo el histórico declarado hasta 2025. Identidad separada: descripción climatológica y posible coeficiente operativo futuro, nunca fuente de métricas TEST ni sustituto de coeficientes TRAIN. IB101 **SHORT_SERIES_DESCRIPTIVE, deployable=false**; IB09 **UNAVAILABLE / Quality FAIL**; IB07 sin validación/test posterior al cierre.

Los contratos [ML](../scientific_pipeline/docs/ml_experiment_contract.md) y [visualización](../scientific_pipeline/docs/visualization_contract.md) conservan su semántica científica congelada. El refinamiento rolling y el runtime ya están fusionados; su implementación no acredita resultados oficiales nuevos. WEB-V0.2 consume los cuatro baselines y AHC publicados sin recalcular ciencia.

### WEB-V0.1 — hito histórico de 25/10/2026

PR [#41](https://github.com/lependens/Capitulo-1/pull/41) publicó FastAPI/DuckDB, React/ECharts, conversor verificado, pruebas y operación Docker Compose; PR [#42](https://github.com/lependens/Capitulo-1/pull/42) corrigió la sintaxis `tmpfs` detectada en el primer arranque. Producción validada en `http://192.168.1.43:8088/`: backend y frontend healthy, enlace exclusivo a LAN, root filesystem de solo lectura y `/data` read-only. Dataset `siar-balears-observational-v1`, 75.496 filas; Parquet SHA256 `6119e3521e8efb1251771a8d6fad2b5a5308e31d7eabeb66e0183c3add0deeca`. Validación funcional de escritorio y móvil PASS.


### WEB-V0.2-AHC — PASS / DEPLOYED

PR [#46](https://github.com/lependens/Capitulo-1/pull/46) y [#47](https://github.com/lependens/Capitulo-1/pull/47) fusionadas. Estado de despliegue confirmado por Dirección 00: [aplicación LAN](http://192.168.1.43:8088/), **LAN only**, montajes de datos **read-only**. Desktop/mobile/API **PASS**. Se mantienen los cuatro baselines y se integran AHC_VALIDATED, **HG_RS_AHC_VALIDATED_SELECTED**, coeficientes y selector **ALL/TRAIN/VALIDATION/TEST**. ALL conserva resultados separados por split, sin fabricar una métrica conjunta. **AHC_CLIMATOLOGY solo operacional**, nunca rendimiento TEST.

`data_recalculated = NO`; `scientific_artifacts_modified = NO`. Derivado **web-derived-v02-ahc-20261031**, SHA256 `875a6dde1fe133e76a03bf2f07f7b51848afe5239c5bc87d9a2f47697cacb360`. Ingestión documentada y snapshot **backup_20261031T160746Z_c8860d65**, **REMOTE_VERIFIED**. Receipt y localizaciones en [DATA_LOCATIONS](DATA_LOCATIONS.md). No se reemplaza la identidad de Dataset v1 por el hash del derivado web.

### ML clásico — runtime implementado, campaña oficial pendiente

**ML-VALIDATION-CONTRACT-REFINEMENT = PASS**, PR [#45](https://github.com/lependens/Capitulo-1/pull/45). Rolling-origin causal en DEVELOPMENT 2004–2020; **Linux smoke = PASS**, prueba de infraestructura sin selección científica ni inspección TEST. [Contrato vigente de validación](../scientific_pipeline/docs/ml_validation_contract_refinement.md).

**Rolling runtime implementation = PASS**, PR [#48](https://github.com/lependens/Capitulo-1/pull/48), merge `87c8040a2f7f8068259f1ff6bdf6bca382b53400`. Comando oficial **ml-rolling-campaign**; STOP estructural de runtime resuelto. Evalúa candidatos en F1–F4, congela selección, refit DEVELOPMENT y una apertura posterior de TEST según el protocolo fusionado. Esto no cambia el protocolo AHC ni sus resultados.

**Campaña ML clásica oficial = PENDING EXECUTION**, pendiente solo de despliegue/ejecución del runtime; no consta cierre científico PASS en el estado comunicado. El smoke y el intento FAILED anterior no son resultados oficiales ni selección de modelos. **ANN = NOT STARTED.** Evidencia operativa y hashes en [DATA_LOCATIONS](DATA_LOCATIONS.md); no se generan modelos ni métricas en esta PR documental.

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

Inventario BACKUP-IMPLEMENTATION-INVENTORY-V1: **capture timer disabled / inactive**. `aemet-forecast-backup.service` static/inactive está disponible, **sin timer independiente**. La inspección no observó hook de backup en el servicio de captura: no se afirma backup automático después de cada intento. Los runs AEMET están también dentro del ámbito de preservación general; esto no demuestra nuevas capturas. Por estar deshabilitada la captura, **new AEMET automatic backup = NOT ACTIVE**. La dependencia vigente es **SIAR-FORECAST-CAPTURE = RUNNING → reactivar AEMET Track A prospectivo**. No necesita SIAR Sync ni esperar a SIAR-FORECAST-PILOT RUNNING. Se conserva raw independiente en `/srv/aemet-forecast/raw/` y cada nueva captura relevante debe quedar cubierta por backup externo.

No están demostrados ET₀, D+n, timezone científico de emisión, recuperación histórica ni entradas suficientes para PM completo. No se infiere timezone ni se integra científicamente con SIAR. **Track B = OPEN**, investigación del feed exacto usado por SIAR, independiente de Track A.

## Storage y preservación

Incidente de storage: **CLOSED_OPERATIONALLY / MONITORING**; **SSD = IN SERVICE**. El parte recibido no aporta evidencia actual de fallo físico activo del Toshiba. SMART PASS; sin sectores reasignados, Reported_Uncorrect, CRC errors ni error log; raíz rw y sin nuevos I/O/ext4 errors observados.

**STORAGE-DATA-PRESERVATION = PASS:** copia independiente fuera del Toshiba, 238/238 archivos de datos size + SHA256 PASS, 12/12 CSV contra POST PASS y 242 archivos totales sin ausencias/discrepancias. Copia descifrada preservada en PC personal. No convierte el incidente cerrado en una línea activa del roadmap.

STOP de infraestructura si reaparecen I/O error, emergency_ro, SMART warning, USB reset asociado o corrupción. Monitorización no equivale a prometer ausencia de fallos futuros.

GitHub conserva código/docs/artefactos versionables. Google Drive con rclone crypt tiene copias verificadas de raw/preservación dentro de sus alcances demostrados; rclone copy, no sync; contenido/nombres/directorios cifrados y restore independiente demostrado. PC personal + OneDrive es copia adicional: no se presenta OneDrive como backup auditado independiente sin verificación expresa. El material crypt/OAuth de recuperación está confirmado OFF-SERVER; no publicar secretos. La cobertura de futuros resultados requiere ingestión explícita.

## Preservación canónica — inventario operativo PASS

**BACKUP-IMPLEMENTATION-INVENTORY-V1 = PASS**, fuente operativa recibida de Dirección 00, sin inspección nueva ni cambios de infraestructura. [Documentación de backup](backup/README.md): servicios/rutas, [cobertura](backup/coverage.md), [restore](backup/restore.md) y [secretos](backup/secrets-and-recovery.md).

`siar-project-preservation.service` static/inactive entre ejecuciones, root:root; `siar-project-preservation.timer` enabled/active, diario **04:15 UTC**, Persistent=true, retraso aleatorio hasta 15 min. Ejecuta snapshot y reconcile para bytes nuevos/modificados. Ingestión event-driven existente; ciencia preservada en `/srv/siar-project/scientific-artifacts`, receipts en `/srv/siar-project/ingestion-receipts`, snapshots y verificaciones en `/srv/siar-project/preservation`. Es distinto del backup SIAR Forecast de 15 min.

**WINDOWS-SCIENCE-INGEST-AUTOMATION = DESIGN_READY / STILL_PENDING**: receptor SFTP + watcher no instalados, provisioning SFTP no autorizado. Artefactos solo Windows no cubiertos hasta ingestión. El inventario no acredita restore específico independiente del namespace SIAR Raw; la evidencia general se conserva separadamente.

## PROJECT-ARTIFACT-BACKUP-V1 — PASS

Parte de cierre de 02 comunicado por Dirección 00: snapshot **backup_20261024T215639Z_63d0ef01**, **REMOTE_VERIFIED**. **128 archivos verificados = 82 scientific-results + 46 SIAR published Raw**. Scientific-results: **1,421,919,603 bytes**. El inventario del servidor tiene **221 entries**: no afirmar que los 221 estén respaldados.

Dataset v1 SHA256 `0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb` y CSV cuatro métodos SHA256 `f55e4ed12691b84dbca4e090c49da538d5e7fad4f2ef371e1307a649cfb1af44`: verificación PASS.

**cryptcheck PASS (128 archivos)** + restore independiente de **2 scientific artifacts y 2 operational references**, hashes PASS. Un intento redundante de verificación por descarga completa fue **INTERRUPTED**: **no declarar full-download verification PASS**. El snapshot se acepta por cryptcheck + restores independientes + SHA256 matches.

Preservación general para artefactos científicos **ya ingeridos al servidor**: **event-driven ingestion PASS, new=0, unchanged=9** en la prueba reportada; **daily preservation reconciliation ENABLED**. Es distinta del reconciliador SIAR Forecast de runs publicados cada **15 minutos**.

**LIMITACIÓN ABIERTA:** futuros resultados solo Windows = **NOT AUTOMATICALLY PROTECTED**. Requieren explicit transfer/ingestion → server preservation area → backup system → Drive crypt. No se garantiza todo resultado futuro ni se confunde inventario con cobertura.

Secret config y credentials **EXCLUDED** del snapshot. Material de recuperación crypt/OAuth **OFF-SERVER**; no publicar rclone.conf, refresh token, OAuth tokens ni crypt passwords. Closeout operativo: `/srv/siar-project/ingestion-receipts/closeout-e0417937.json`, referencia sin copiar contenido.

Toshiba ya no es la única copia **para los artefactos científicos y SIAR published Raw cubiertos por el snapshot verificado**. GitHub sigue siendo autoridad versionable, no copia íntegra de outputs masivos. Esta misión backup **no cambió Forecast**: timer captura y reconciliador ACTIVE; nuevo run automático real con backup/COMPLETE remoto aún pendiente. AEMET disabled/inactive, captures=0, gate=CLOSED, sin nuevos raw.

## Próximos pasos

1. **05 / ciencia:** desplegar/ejecutar `ml-rolling-campaign` y publicar la campaña ML clásica oficial con protocolo, hashes y evidencia de cierre. Runtime y Linux smoke ya PASS; no repetir su implementación como si faltara. ANN después de evaluar ML.
2. **06 / producto:** conservar WEB-V0.2-AHC desplegada y consumir outputs ML oficiales publicados para WEB-V0.3-ML; no recalcular ciencia desde UI/API. Forecast se integra posteriormente bajo sus contratos.
3. **02B / preservación:** ingerir y preservar cada nuevo artefacto publicado. WEB-V0.2 ya tiene receipt y snapshot REMOTE_VERIFIED; Windows automation sigue DESIGN_READY / STILL_PENDING.
4. Infraestructura: gate SIAR Forecast de nuevo run real con backup/COMPLETE, AEMET tras CAPTURE RUNNING y SIAR Sync alignment/incremental 2026 permanecen separados, sin cambios en esta misión.
5. Issue/valid time/D+n, ET₀+Kc/ETc/riego y validación prospectiva mantienen sus gates propios.

**05 produce ciencia → 06 consume ciencia publicada → 02B preserva nuevos artefactos.** MILESTONES conserva la historia; DATA_LOCATIONS identifica bytes/receipts; [backup canónico](backup/README.md) delimita cobertura. Se escalan arquitectura, seguridad, integridad raw y semántica científica.

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
