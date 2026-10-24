# PROJECT STATUS — SIAR / ET₀

**Actualizado: 2026-10-24 (Europe/Madrid).** Fuente principal del estado vigente. Actualización científica sobre main `a0b440fa3ab92e4593be521e4e819c6421327656`; estado operativo según el último parte de Dirección 00 recibido en esta misión. Un PASS histórico no acredita que un timer siga ejecutándose hoy.

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
| PM/HS v1 | Primera ejecución oficial PASS, publicada mediante PR #31 | Interpretar cohortes y limitaciones; no calibrar para reproducir EtPMon |
| Comparación clásica | PM/HS/VAL/HG-Rs oficiales PASS; CSV verificado | Preservación y límites; sin calibración |
| SIAR Sync | v2.3.3 canónica; SIAR-SYNC-RUNTIME = HOLD | SIAR-SYNC-PRODUCTION-ALIGNMENT + SIAR-SYNC-INCREMENTAL-2026 |
| SIAR Forecast | Collector, runtime, raw y backup/restore demostrados; timers pausados tras incidente según último parte | Un ciclo automático real → CAPTURE RUNNING; al menos dos → PILOT RUNNING |
| AEMET Track A | E2E DEMONSTRATED; pausado tras incidente según último parte | Reactivación prospectiva después de SIAR-FORECAST-CAPTURE RUNNING; no depende de SIAR Sync |
| AEMET Track B | OPEN | Investigar feed AEMET exacto utilizado por SIAR, independientemente de Track A |
| Storage Toshiba | CLOSED_OPERATIONALLY / MONITORING; preservación PASS | STOP de infraestructura ante señales nuevas verificadas |
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

**Actualización posterior al punto de partida del encargo:** [PR #30](https://github.com/lependens/Capitulo-1/pull/30) resolvió el redondeo VPD de tres fechas IB10; [PR #31](https://github.com/lependens/Capitulo-1/pull/31) publicó **PM-VPD-FLOAT-TOLERANCE-AND-BASELINES = PASS**. Ya no se trata de una ejecución pendiente.

La tolerancia es absoluta 1e-12 kPa, relativa 0.0, solo para exceso ea>es con presiones finitas y no negativas; conserva rechazo fuera del margen y VPD positivos exactos. No altera datos, humedad, es, ea ni ecuaciones. Pipeline ejecutado: `39c101e5999245eb15e3c6c1c867d90b4ac44468`. Ver [supuestos](../scientific_pipeline/docs/assumptions.md).

PM calculó 70.896 predicciones y HS 70.979, con 0 fallos. Evaluación: PM 70.861 pares propios; HS 70.912; intersección 70.861. Se conservan 47 pares con EtPMon=0. Las [métricas oficiales y sus desgloses](../scientific_pipeline/results/pm-vpd-20261023T194553Z/README.md) son la autoridad de resultados; no se vuelven a calcular en esta misión.

EtPMon es referencia calculada SIAR, no medida independiente de ET₀ real. Permanecen UNITS_PROVENANCE_PARTIAL, viento a 2 m por convención sin verificación instrumental exhaustiva, exclusión IB09 por FAIL, cohortes desiguales y muestra corta IB101. PM/HS son baselines sin ML, AHC ni calibración. VALIANTZAS_V1 y comparación clásica PASS el 24/10; sus resultados antiguos permanecen legacy. [Métricas y CSV](../scientific_pipeline/results/classical-20261024T034529Z/README.md).

## SIAR Sync: HOLD independiente del snapshot científico

SIAR Sync v2.3.3 mantiene autenticación dinámica, bloques ≤7 días, esquema 23 columnas, escritura atómica, reintentos HTTP, persistencia local ante fallos Git y checkpoints. SIAR_CA_BUNDLE es opcional y la verificación TLS permanece activa; no se usa verify=False. INC-INFRA-001 está resuelta e init: true es canónico.

**SIAR-SYNC-RUNTIME = HOLD.** Diagnóstico de 02 comunicado por Dirección 00: checkout residual `quality/recovery-20261012T085131Z`, HEAD `18e6d893...`; main ya contiene la recuperación publicada. El worker tiene target default 2025-12-31 y no hay modo incremental 2026 demostrado que vaya de último dato útil a actualidad sin reabrir huecos históricos. /start podría intentar aproximadamente 1.958 huecos históricos de activas.

El panel mezcla historical gaps con current/prospective lag y no aplica correctamente Fecha_Baja a cerradas, aunque el worker sí. Reactivar exige **SIAR-SYNC-PRODUCTION-ALIGNMENT + SIAR-SYNC-INCREMENTAL-2026**; no ejecutar /start como actualización rutinaria. Este documento no despliega ni actualiza el checkout del servidor.

## Forecast SIAR: adquisición prospectiva desacoplada

Collector independiente en `forecast_collector/`, runtime permanente `/opt/siar-forecast`, configuración `/etc/siar-forecast`, raw `/srv/siar-forecast/raw/`, identidad dedicada siar-forecast. SMOKE-FC-001, BACKUP-FC-001 y FC-STATIONS-001 PASS. Append-only, hashes, schemas, TLS, publicación atómica y restore independiente demostrados.

Ocho estaciones funcionales: IB01, IB02, IB03, IB04, IB05, IB06, IB10, IB11. IB08 sigue **ACTIVE / TEMPORARILY_UNAVAILABLE_EXTERNAL** por indisponibilidad externa observada en su combinación de catálogo; no bloquea las demás ni se transforma en CLOSED. No se conoce la causa interna exacta de SIAR/MAPA.

Según el último parte, los timers quedaron **pausados tras el incidente**. Forecast puede reactivarse aunque SIAR Sync siga HOLD. No reconstruir emisiones perdidas durante la pausa.

- **PENDIENTE:** 1 ciclo automático real demostrado → SIAR-FORECAST-CAPTURE = RUNNING.
- **PENDIENTE:** mínimo 2 ciclos automáticos demostrados → SIAR-FORECAST-PILOT = RUNNING.

No se afirma que esos gates hayan pasado sin un parte posterior verificable. La cadencia experimental aprobada es 08:00 / 14:00 / 20:00 Europe/Madrid. source issue timestamp puede permanecer null; D+0…D+6 no está asignado. HTML puede ser persisted, omitted_sensitive o not_available; las evidencias permitidas se conservan según contrato.

## AEMET: Track A y Track B

Track A E2E DEMONSTRATED; PR #18 Collector, #19 comparación id municipal, #20 permisos POSIX. Wrapper y payload raw append-only con bytes, timestamps y provenance preservados. IB04 Son Ferriol → Palma → municipio 07040 representa la capital municipal, no la estación exacta.

Último parte: piloto **pausado tras el incidente**. La dependencia vigente es **SIAR-FORECAST-CAPTURE = RUNNING → reactivar AEMET Track A prospectivo**. No necesita SIAR Sync ni esperar a SIAR-FORECAST-PILOT RUNNING. Se conserva raw independiente en `/srv/aemet-forecast/raw/` y cada nueva captura relevante debe quedar cubierta por backup externo.

No están demostrados ET₀, D+n, timezone científico de emisión, recuperación histórica ni entradas suficientes para PM completo. No se infiere timezone ni se integra científicamente con SIAR. **Track B = OPEN**, investigación del feed exacto usado por SIAR, independiente de Track A.

## Storage y preservación

Toshiba: **CLOSED_OPERATIONALLY / MONITORING**, sin evidencia actual de fallo físico activo en el parte recibido. SMART PASS; sin sectores reasignados, Reported_Uncorrect, CRC errors ni error log; raíz rw y sin nuevos I/O/ext4 errors observados.

**STORAGE-DATA-PRESERVATION = PASS:** copia independiente fuera del Toshiba, 238/238 archivos de datos size + SHA256 PASS, 12/12 CSV contra POST PASS y 242 archivos totales sin ausencias/discrepancias. Copia descifrada preservada en PC personal. No convierte el incidente cerrado en una línea activa del roadmap.

STOP de infraestructura si reaparecen I/O error, emergency_ro, SMART warning, USB reset asociado o corrupción. Monitorización no equivale a prometer ausencia de fallos futuros.

GitHub conserva código/docs/artefactos versionables. Google Drive con rclone crypt cubre raw operativo y preservación; rclone copy, no sync; contenido/nombres/directorios cifrados y restore independiente demostrado. PC personal + OneDrive es copia adicional: no se presenta OneDrive como backup auditado independiente sin verificación expresa. La recuperabilidad externa del material crypt y la cobertura de nuevos raw mantienen validaciones propias.

## Próximos gates y gobierno

1. 02: reactivar Forecast SIAR, demostrar un ciclo automático y después dos; habilitar AEMET tras CAPTURE RUNNING, sin esperar SIAR Sync.
2. 02: resolver alignment y modo incremental 2026 de SIAR Sync antes de levantar HOLD.
3. Ciencia: preservar comparación PM/HS/VAL/HG-Rs y CSV publicados; retener limitaciones y definir partición temporal antes de cualquier calibración o ML.
4. Después: modelos estadísticos, árboles, boosting y ANN con split temporal sin leakage.
5. Forecast: acumular emisiones, resolver issue/valid time/D+n y datasets derivados antes de comparar con observaciones posteriores.
6. Producto Web/API, ET₀+Kc, ETc y riego después de contratos y validaciones.

Los workstreams trabajan autónomamente dentro de objetivos, invariantes y STOP fijados por dirección. Resuelven errores operativos localmente y escalan arquitectura, seguridad, integridad raw o semántica científica. Una evidencia nueva modifica el estado maestro con su fuente; no se infieren resultados operativos por la mera existencia de código o timers.


## Comparación clásica oficial — 24/10/2026

[Resultado, cohortes, métricas, CSV y hashes](../scientific_pipeline/results/classical-20261024T034529Z/README.md).
Dataset v1 intacto; PM/HS oficiales reutilizados; VAL 70.944 predicciones,
70.878 evaluables, 0 fallos. Martí Ecs.1b+3 con 0.408, sin calibración.
Pipeline 4ac52e25c803cb2301316c5cb1353bf30cdb427a; ambas identidades audit/canonical preservadas.


## Ampliación HG-Rs — 24/10/2026

[Comparación de cuatro modelos y hashes](../scientific_pipeline/results/four-classical-20261024T042250Z/README.md).
HG_RS_V1 70978 calculadas, 70912 evaluables, 0 fallos;
cohorte común 70860. PR #35, pipeline `a0b440fa3ab92e4593be521e4e819c6421327656`.
Dataset y todos los resultados PM/HS/VAL anteriores intactos, sin calibración.
75 tests/CI PASS. Ambos SHA audit/canonical preservados.
