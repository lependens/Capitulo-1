# ROADMAP — SIAR / ET₀

**Actualizado: 2026-10-31 (Europe/Madrid), consolidación hasta PR #49.** Próximos gates y dependencias; estado demostrado en [PROJECT_STATUS](PROJECT_STATUS.md), hitos en [MILESTONES](MILESTONES.md) y ubicaciones en [DATA_LOCATIONS](DATA_LOCATIONS.md).

## Leyenda y trabajo paralelo

✅ demostrado; 🟠 pendiente de validación o investigación; ⚪ futuro. Histórico y superado se fechan explícitamente. Un resultado Quality COMPLETE no implica observaciones para el 100 % de días.

| Línea | Ya demostrado | Siguiente gate |
|---|---|---|
| Observaciones / Quality | POST corregido COMPLETE; 2.345 ausencias confirmadas, 0 PENDING_RETRY | Mantener snapshot cerrado; CATALOG-TZ-001 no bloqueante |
| Ciencia / 05 | Dataset, cuatro baselines, AHC, rolling validation, Linux smoke y rolling runtime PASS | Ejecutar/publicar campaña ML clásica oficial; ANN después de evaluación ML |
| Preservación / 02 | PROJECT-ARTIFACT-BACKUP-V1 PASS; 128 archivos remotos verificados | Ingestión explícita de futuros artefactos Windows; seguimiento continuo |
| SIAR Sync / 02 | v2.3.3 canónica, snapshot científico 2025 cerrado | HOLD → production alignment + incremental 2026 demostrado |
| Forecast SIAR / 02 | Timer ACTIVE; reconciliador enabled/active cada 15 min; OAuth Production | Nuevo run automático → backup automático → COMPLETE remoto → CAPTURE RUNNING; mínimo 2 ciclos PILOT RUNNING |
| AEMET Track A / 02 | E2E DEMONSTRATED; capture disabled/inactive; nuevo backup automático NOT ACTIVE | Reactivar tras SIAR-FORECAST-CAPTURE RUNNING, sin dependencia de Sync |
| AEMET Track B | Investigación independiente OPEN | Identificar feed exacto usado por SIAR |
| Producto / 06 | WEB-V0.2-AHC PASS / DEPLOYED en LAN; WEB-V0.1 histórico | WEB-V0.3-ML espera outputs ML oficiales publicados; Forecast después |

**La captura prospectiva no espera a los modelos ni a reactivar SIAR Sync.** No reconstruir emisiones perdidas durante la pausa. El incidente Toshiba está CLOSED_OPERATIONALLY / MONITORING; no es una nueva fase activa.

## Fase 0 — Base y separación del monorepo

- ✅ SIAR Sync, collectors SIAR/AEMET y scientific_pipeline separados.
- ✅ PR #9/#10: código y derivados históricos bajo legacy; datos_siar_baleares solo CSV canónicos.
- ✅ Documentación maestra y bitácoras con historia diferenciada del estado vigente.
- ✅ INC-INFRA-001 resuelta; init: true canónico. No quedan observaciones de zombies como tarea abierta.
- Mantener código, raw, artefactos Quality y resultados con responsabilidades diferenciadas.

## Fase 1 — Observaciones y Quality

✅ Cierre de Quality, no R2 en ejecución. POST vigente **SIAR-QA-POST-20261020T193140Z**, superseding del original **SIAR-QA-POST-20261018T233901Z**, ambos preservados. PR #29 corrige ledger sin alterar raw.

12 estaciones: 9 ACTIVE, 3 CLOSED; 75.496 filas raw, 75.495 dentro de periodo oficial; 77.840 días esperados; cobertura 96.987410 %. Ledger 2.346 = 1 RECOVERED + 2.345 CONFIRMED_NO_DATA, 0 PENDING_RETRY.

Audit completeness COMPLETE; observational completeness INCOMPLETE por ausencias SIAR confirmadas. IB07/2012-11-13 RECOVERED con evidencia F2 y 23/23 campos; IB09/2005-10-23 raw preservado OUTSIDE_OFFICIAL_PERIOD. CATALOG-TZ-001 OPEN / NON_BLOCKING, sin autocorrección del catálogo.

🟠 **Reactivación futura separada:** SIAR-SYNC-RUNTIME HOLD hasta SIAR-SYNC-PRODUCTION-ALIGNMENT y SIAR-SYNC-INCREMENTAL-2026. Demostrar último dato útil → actualidad sin reabrir indiscriminadamente huecos históricos; separar historical gaps de lag actual en panel y aplicar Fecha_Baja correctamente en su presentación. No reabrir la adquisición científica 2025 para cumplir ese gate.

SIAR Sync España sigue siendo futuro, sin implementación actual.

## Fase 2 — Forecast SIAR prospectivo

✅ Collector append-only; sesión/CSRF/cálculo/exportZIP/CSV; guard rail de secretos, HTML sensible omitible, ZIP seguro, SHA-256, schemas y publicación atómica. Runtime permanente /opt + /etc + /srv/siar-forecast/raw con identidad dedicada. SMOKE-FC-001, BACKUP-FC-001 y FC-STATIONS-001 PASS; multi-smoke 8 completas de 9 intentadas.

Último parte 24/10: **timer SIAR Forecast ACTIVE** y reconciliador **enabled/active cada 15 minutos**, último Result success. OAuth Production + reauthorization/crypt listing PASS elimina el límite específico de siete días de Testing; no garantiza tokens eternos. Copy to crypt + restore independiente + SHA256 MATCH demostrados.

1. 🟠 Demostrar **nuevo run real automático → backup automático → COMPLETE remoto**; primer ciclo completo PASS → SIAR-FORECAST-CAPTURE RUNNING.
2. 🟠 Al menos dos ciclos automáticos PASS → SIAR-FORECAST-PILOT RUNNING.
3. 🟠 Mantener cobertura de backup de nuevos raw y recuperabilidad externa.
4. 🟠 Acumular observación operacional y validar cadencia/semántica sin asignar D+n.

Reconciliador: rclone copy (no sync), solo runs publicados pendientes en /srv/siar-forecast/raw/runs/, con estados complete/partial/failed. Excluye raw/tmp, CSV Sync, código, /etc completo y resto del servidor. No satisface por sí solo PROJECT-ARTIFACT-BACKUP-V1 ni acredita el ciclo real todavía pendiente.

Target funcional: IB01, IB02, IB03, IB04, IB05, IB06, IB10, IB11. IB08 sigue ACTIVE / TEMPORARILY_UNAVAILABLE_EXTERNAL, vigilada aparte; no bloquea las ocho ni se etiqueta CLOSED. Cadencia experimental 08:00 / 14:00 / 20:00 Europe/Madrid.

source issue time puede ser null; D+0…D+6 y recuperación histórica de emisiones no están demostrados. No rellenar la pausa retrospectivamente. Salida de fase: captura y backup sostenidos con evidencia; un PASS de smoke no equivale al gate periódico.

## Fase 3 — AEMET

✅ Track A municipal diario OpenData: E2E DEMONSTRATED, estabilización PASS. PR #18/#19/#20: collector, id municipal y permisos POSIX. Conserva wrapper.raw y payload.raw sin modificar bytes; id 7040 se compara canónicamente con 07040 sin editar raw.

Piloto IB04 → Palma → municipio 07040, capital municipal, no punto exacto de estación. Último parte: **capture timer disabled/inactive**. El servicio específico de backup está disponible, static/inactive, sin timer independiente. El inventario no observó hook de backup en el servicio de captura; no se afirma ejecución automática tras cada intento. Los runs AEMET están en el ámbito separado de preservación general. Con captura deshabilitada, **new AEMET automatic backup = NOT ACTIVE**.

**Dependencia vigente:** SIAR-FORECAST-CAPTURE RUNNING → reactivar AEMET Track A. Se supera la dependencia antigua de esperar SIAR-FORECAST-PILOT RUNNING; nunca depende de SIAR Sync.

🟠 Validar reactivación, runtime/configuración, raw independiente /srv/aemet-forecast/raw, backup rclone crypt, restore y scheduler experimental con evidencia de los nuevos ciclos. No se infiere ejecución actual del mero contrato del collector.

ET₀ NO; D+n NO; timezone de emisión inferida NO; integración científica SIAR/AEMET NO; histórico de emisiones no demostrado. El producto no aporta por sí solo entradas suficientes para FAO-56 PM completo.

🟠 **Track B OPEN:** investigar feed AEMET exacto utilizado por SIAR, independiente de Track A.

## Fase 4 — Dataset Observational v1

✅ Construido y validado estructuralmente con POST corregido y handoff autorizado. Dataset siar-balears-observational-v1; SHA256 **0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb**.

Conserva 75.496 observaciones raw, 23 columnas, una fila exterior, ledger y provenance; no interpola, imputa ni fabrica meteorología. El JSON completo tiene backup externo verificado en el snapshot aceptado; no está íntegro en Git. Referencias verificables en [ubicaciones](DATA_LOCATIONS.md).

🟠 Mantener preservación, identidad y acceso; revisar convención de resultados con el productor antes de mover archivos. scientific-results/observational-v1/{dataset,baselines,exports} es solo una propuesta, no un despliegue ni un movimiento aprobado.

## Fase 5 — Cuatro baselines clásicos oficiales: PASS

✅ PM_FAO56_V1, HS_RA_V1, VALIANTZAS_V1 y HG_RS_V1 implementados y oficiales, sin calibración local. PR #30/#31 PM/HS, #33/#34 Valiantzas y tres métodos, #35/#36 HG-Rs y cuatro métodos. Dataset/raw/Quality intactos. Pipeline HG-Rs `a0b440fa3ab92e4593be521e4e819c6421327656`; publicación `9f3dde8e22ab8e8f76aeaa273b5e7681558ebbcf`. **75 tests y CI 3.11/3.12/3.13 PASS**.

[Resultados y contrato de comparación](../scientific_pipeline/results/four-classical-20261024T042250Z/README.md): cohorte común PM_HS_VAL_HGRS n=70.860, cohortes propias, pares/triples y desgloses. No queda implementar Valiantzas/HG-Rs como siguiente tarea.

HG-Rs tiene mejor RMSE/R² que VAL y VAL mejor MAE/bias en la cohorte común; no hay vencedor universal. La mayor similitud de inputs con EtPMon puede favorecer concordancia, sin causalidad demostrada. Separar accuracy against EtPMon de input requirements / operational availability; sigue importando precisión con menos variables. EtPMon no es ET₀ instrumental independiente.

## Fase 6 — AHC cerrado, runtime ML rolling listo y campaña pendiente

✅ **AHC-METHODOLOGY-AND-TEMPORAL-VALIDATION PASS**, PR #38/#39. OLS0 por estación a=Σ(x·PM)/Σ(x²), referencia primaria PM_FAO56_V1; EtPMon secundaria calculada. TRAIN 2004–2015, VALIDATION 2016–2020, TEST 2021–2025; ajuste solo TRAIN, sin refit TRAIN+VALIDATION. TEST retrospective_locked, no prospectivo nunca visto.

✅ VALIDATION eligió HS annual, HG-Rs seasonal y VAL annual; mensual no justifica complejidad extra. No reseleccionar tras TEST. AHC_CLIMATOLOGY se calculó después de TEST con todo el histórico hasta 2025 y permanece separado de métricas fuera de muestra. IB101 descriptivo corto, deployable=false; IB09 UNAVAILABLE/FAIL. Estabilidad PASS no garantiza mejora todos los años; conservar empeoramientos.

✅ **PROJECT-ARTIFACT-BACKUP-V1 PASS**, snapshot REMOTE_VERIFIED de 128 archivos; no es un gate pendiente. Daily reconciliation ENABLED y event-driven ingestion PASS para material ya ingerido al servidor. Futuros resultados solo Windows requieren transferencia explícita; no están automáticamente protegidos.

✅ [ML contract](../scientific_pipeline/docs/ml_experiment_contract.md) **FROZEN / READY FOR FUTURE EXPERIMENTS**; [visualization contract](../scientific_pipeline/docs/visualization_contract.md) **FROZEN / AVAILABLE TO 06**.

Estado científico y de implementación:

- ✅ Dataset Observational v1.
- ✅ Cuatro baselines clásicos oficiales.
- ✅ AHC temporal oficial.
- ✅ Rolling validation contract (PR #45).
- ✅ Linux smoke, solo infraestructura; sin selección científica ni TEST.
- ✅ Runtime rolling oficial (PR #48), comando `ml-rolling-campaign`.
- 🟠 Campaña ML clásica oficial: PENDING EXECUTION; siguiente gate despliegue/ejecución y publicación verificable, no implementar de nuevo el runtime.
- ⚪ ANN NOT STARTED; después de evaluar ML clásico.

[Protocolo rolling vigente](../scientific_pipeline/docs/ml_validation_contract_refinement.md): selección en DEVELOPMENT F1–F4, congelación, refit DEVELOPMENT y TEST posterior una vez. Conservar `retrospective_locked`; no aplicar este refit al AHC ya publicado. El intento FAILED y el smoke no sirven para seleccionar ciencia. No reabrir baselines ni metodología AHC.

**05 produce ciencia; 06 consume resultados publicados; 02B preserva nuevos artefactos.** WEB-V0.3-ML espera outputs oficiales, no convierte disponibilidad de runtime en campaña completada.

## Fase 7 — Evaluación predictiva

⚪ Acumular raw SIAR y AEMET prospectivamente → resolver issue time / valid time / D+n y correspondencia estación/municipio → datasets forecast derivados → comparar con observaciones posteriores.

No asignar horizonte por posición de fila ni timezone por conjetura. No mezclar forecast municipal con estación puntual sin explicitar correspondencia y límites. No esperar a terminar los modelos para comenzar la acumulación prospectiva.

## Fase 8 — Producto

✅ **WEB-V0.1 histórico**, PR #41/#42: primer despliegue LAN de consulta read-only de cuatro baselines.

✅ **WEB-V0.2-AHC PASS / DEPLOYED**, PR #46/#47. Cuatro baselines, AHC_VALIDATED, HG_RS_AHC_VALIDATED_SELECTED, coeficientes y ALL/TRAIN/VALIDATION/TEST visibles; climatología solo operacional. Desktop/mobile/API PASS; [LAN](http://192.168.1.43:8088/), data mounts read-only, sin recalcular datos ni modificar ciencia. Derivado y backup REMOTE_VERIFIED identificados en [DATA_LOCATIONS](DATA_LOCATIONS.md).

🟠 **WEB-V0.3-ML:** espera publicación de outputs oficiales de la campaña ML. 06 consume tablas/manifests publicados por 05; 02B preserva los nuevos artefactos. No entrenar modelos ni recalcular ciencia desde producto.

⚪ Integración Forecast posterior con semántica issue/valid time y D+n demostrada. Después ET₀ + Kc → ETc → necesidades de riego. Mantener separados adquisición, ciencia, modelado y producto; prototipos legacy no son resultados v1.

## Infraestructura, backups y STOP

✅ **BACKUP-IMPLEMENTATION-INVENTORY-V1 PASS**: implementación existente canonizada en [docs/backup](backup/README.md). Preservación general diaria 04:15 UTC (Persistent, jitter hasta 15 min) e ingestión event-driven; backup SIAR cada 15 min; AEMET sin timer propio ni hook de captura observado. No cambia ninguno de los gates de captura.

🟠 **WINDOWS-SCIENCE-INGEST-AUTOMATION = DESIGN_READY / STILL_PENDING**: desarrollar/provisionar receptor y watcher y validar E2E sintético en misión independiente. [Diseño](backup/windows-ingestion.md) no desplegado; provisioning SFTP no autorizado aquí. Hasta ingestión explícita, Windows-only no está cubierto. Mantener verificación de cobertura, receipts y recuperación off-server; no asumir backup masivo de workspaces, /etc ni servidor completo.

STORAGE-DATA-PRESERVATION PASS; incidente CLOSED_OPERATIONALLY / MONITORING; **SSD IN SERVICE**. No crear un roadmap de sustitución del SSD; reabrir solo con nueva evidencia objetiva. 238/238 archivos de datos size/SHA256 y 12/12 CSV contra POST PASS; 242 archivos totales sin ausencias/discrepancias. No se reabre como proyecto activo por mantener monitorización.

STOP ante nueva evidencia de I/O error, emergency_ro, SMART warning, USB reset asociado o corrupción. Mantener GitHub para código/docs/artefactos versionables, Drive cifrado para raw/preservación y copia adicional PC/OneDrive. OneDrive no se etiqueta backup auditado independiente sin prueba expresa.

**PROJECT-ARTIFACT-BACKUP-V1 PASS:** snapshot backup_20261024T215639Z_63d0ef01 REMOTE_VERIFIED; 82 archivos científicos (1,421,919,603 bytes) + 46 SIAR published Raw = 128 verificados. 221 inventariados no significa 221 respaldados. cryptcheck + restores independientes + hashes PASS; descarga completa redundante INTERRUPTED, no full-download PASS. Secret config/credentials excluidos; crypt/OAuth OFF-SERVER.

Event-driven ingestion PASS (new=0, unchanged=9) y daily reconciliation ENABLED protegen artefactos ya ingeridos al servidor; futuros resultados solo Windows necesitan transferencia/ingestión. El reconciliador SIAR de 15 min es distinto y no cambia el gate de nuevo run automático con backup/COMPLETE remoto. Cada nuevo raw prospectivo relevante requiere backup externo. Mantener rclone copy, no sync; validar restore y material crypt recuperable fuera del servidor. El PASS histórico de backup no acredita que una nueva captura ya tenga copia.

## Gobierno y actualización

Los workstreams resuelven problemas operativos dentro de objetivos/invariantes/STOP. Se escalan arquitectura, seguridad, integridad raw y semántica científica. Cambios de estado requieren evidencia y actualización de PROJECT_STATUS; MILESTONES se amplía sin borrar hitos previos. Las fases 2/3 permanecen abiertas y no se declara un RUNNING nuevo sin parte consolidado.


## Ampliación clásica completada — 24/10/2026

✅ HG_RS_V1 con Rs medida incorporado y ejecutado sin calibración;
[comparación oficial de cuatro métodos y CSV](../scientific_pipeline/results/four-classical-20261024T042250Z/README.md), 75 tests/CI PASS.
PM/HS/VAL y Dataset v1 intactos. AHC y backup ya están PASS; siguientes experimentos ML v1 bajo contrato congelado, con preservación continua y consumo por 06 sin recalcular ciencia.
