# ROADMAP — SIAR / ET₀

**Actualizado: 2026-10-24 (Europe/Madrid).** Próximos gates y dependencias; estado demostrado en [PROJECT_STATUS](PROJECT_STATUS.md), hitos en [MILESTONES](MILESTONES.md) y ubicaciones en [DATA_LOCATIONS](DATA_LOCATIONS.md).

## Leyenda y trabajo paralelo

✅ demostrado; 🟠 pendiente de validación o investigación; ⚪ futuro. Histórico y superado se fechan explícitamente. Un resultado Quality COMPLETE no implica observaciones para el 100 % de días.

| Línea | Ya demostrado | Siguiente gate |
|---|---|---|
| Observaciones / Quality | POST corregido COMPLETE; 2.345 ausencias confirmadas, 0 PENDING_RETRY | Mantener snapshot cerrado; CATALOG-TZ-001 no bloqueante |
| Ciencia | Dataset v1 y cuatro baselines oficiales PASS | PROJECT-ARTIFACT-BACKUP-V1 → AHC-METHODOLOGY-REVIEW → temporal evaluation contract |
| SIAR Sync / 02 | v2.3.3 canónica, snapshot científico 2025 cerrado | HOLD → production alignment + incremental 2026 demostrado |
| Forecast SIAR / 02 | Timer ACTIVE; reconciliador enabled/active cada 15 min; OAuth Production | Nuevo run automático → backup automático → COMPLETE remoto → CAPTURE RUNNING; mínimo 2 ciclos PILOT RUNNING |
| AEMET Track A / 02 | E2E DEMONSTRATED; capture disabled/inactive; nuevo backup automático NOT ACTIVE | Reactivar tras SIAR-FORECAST-CAPTURE RUNNING, sin dependencia de Sync |
| AEMET Track B | Investigación independiente OPEN | Identificar feed exacto usado por SIAR |

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

Piloto IB04 → Palma → municipio 07040, capital municipal, no punto exacto de estación. Último parte: **capture timer disabled/inactive**. El backup se ejecuta tras cada intento de captura, success o failure; no existe timer independiente. Con captura deshabilitada, **new AEMET automatic backup = NOT ACTIVE**.

**Dependencia vigente:** SIAR-FORECAST-CAPTURE RUNNING → reactivar AEMET Track A. Se supera la dependencia antigua de esperar SIAR-FORECAST-PILOT RUNNING; nunca depende de SIAR Sync.

🟠 Validar reactivación, runtime/configuración, raw independiente /srv/aemet-forecast/raw, backup rclone crypt, restore y scheduler experimental con evidencia de los nuevos ciclos. No se infiere ejecución actual del mero contrato del collector.

ET₀ NO; D+n NO; timezone de emisión inferida NO; integración científica SIAR/AEMET NO; histórico de emisiones no demostrado. El producto no aporta por sí solo entradas suficientes para FAO-56 PM completo.

🟠 **Track B OPEN:** investigar feed AEMET exacto utilizado por SIAR, independiente de Track A.

## Fase 4 — Dataset Observational v1

✅ Construido y validado estructuralmente con POST corregido y handoff autorizado. Dataset siar-balears-observational-v1; SHA256 **0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb**.

Conserva 75.496 observaciones raw, 23 columnas, una fila exterior, ledger y provenance; no interpola, imputa ni fabrica meteorología. El JSON completo es un artefacto local preservado, no se afirma almacenado íntegro en Git. Referencias verificables en [ubicaciones](DATA_LOCATIONS.md).

🟠 Mantener preservación, identidad y acceso; revisar convención de resultados con el productor antes de mover archivos. scientific-results/observational-v1/{dataset,baselines,exports} es solo una propuesta, no un despliegue ni un movimiento aprobado.

## Fase 5 — Cuatro baselines clásicos oficiales: PASS

✅ PM_FAO56_V1, HS_RA_V1, VALIANTZAS_V1 y HG_RS_V1 implementados y oficiales, sin calibración local. PR #30/#31 PM/HS, #33/#34 Valiantzas y tres métodos, #35/#36 HG-Rs y cuatro métodos. Dataset/raw/Quality intactos. Pipeline HG-Rs `a0b440fa3ab92e4593be521e4e819c6421327656`; publicación `9f3dde8e22ab8e8f76aeaa273b5e7681558ebbcf`. **75 tests y CI 3.11/3.12/3.13 PASS**.

[Resultados y contrato de comparación](../scientific_pipeline/results/four-classical-20261024T042250Z/README.md): cohorte común PM_HS_VAL_HGRS n=70.860, cohortes propias, pares/triples y desgloses. No queda implementar Valiantzas/HG-Rs como siguiente tarea.

HG-Rs tiene mejor RMSE/R² que VAL y VAL mejor MAE/bias en la cohorte común; no hay vencedor universal. La mayor similitud de inputs con EtPMon puede favorecer concordancia, sin causalidad demostrada. Separar accuracy against EtPMon de input requirements / operational availability; sigue importando precisión con menos variables. EtPMon no es ET₀ instrumental independiente.

## Fase 6 — Preservación, AHC y evaluación antes de modelos nuevos

Orden vigente:

1. 🟠 **PROJECT-ARTIFACT-BACKUP-V1:** política explícita para Dataset Observational v1 completo, handoff, scientific-results completos, CSV exports, predictions, manifests, future ML artifacts, raw Forecast y configuración operativa recuperable. Misión separada a 02; no se afirma que ya estén respaldados. GitHub no es copia completa del proyecto científico.
2. 🟠 **AHC-METHODOLOGY-REVIEW:** revisar bibliografía; distinguir coeficiente climatológico de estación de calibración predictiva; estudiar estabilidad temporal y AHC anual/estacional/mensual si procede.
3. 🟠 **Temporal evaluation contract:** definir objetivos, datos de ajuste, particiones y validación temporal antes de AHC_V1 o cualquier modelo calibrado. No decidir en esta PR si AHC usa toda la serie o solo TRAIN; se resuelve en la misión científica específica.
4. ⚪ Modelos calibrados simples.
5. ⚪ Regresión / árboles / boosting.
6. ⚪ ANN después, sin salto inmediato desde los cuatro baselines.

AHC estimado de datos es un parámetro calibrado. El AHC legacy mean(ET0_calc/ET0_empirical), calculado por estación sobre toda la serie y evaluado en los mismos días, es **HISTÓRICO / full-series / in-sample**; no demuestra validación predictiva. No cerrar todavía AHC v1 ni reetiquetar coeficientes históricos.

El contrato futuro debe prevenir leakage y distinguir uso climatológico/descriptivo de evaluación predictiva. Las reglas de aprendizaje de ML continúan exigiendo separación temporal y ajuste de transformaciones sin usar evaluación; su aplicación específica a AHC queda pendiente de la revisión.

Salida: comparación reproducible de precisión, cobertura y requisitos de inputs, no una mejora agregada sin cohorte. Retener IB09 FAIL, IB101 corto y procedencia instrumental parcial como limitaciones.

## Fase 7 — Evaluación predictiva

⚪ Acumular raw SIAR y AEMET prospectivamente → resolver issue time / valid time / D+n y correspondencia estación/municipio → datasets forecast derivados → comparar con observaciones posteriores.

No asignar horizonte por posición de fila ni timezone por conjetura. No mezclar forecast municipal con estación puntual sin explicitar correspondencia y límites. No esperar a terminar los modelos para comenzar la acumulación prospectiva.

## Fase 8 — Producto

⚪ Web/API → ET₀ + Kc → ETc → necesidades de riego. Separar adquisición, procesamiento científico, modelado y producto. Los prototipos ANN y dashboards legacy conservan valor histórico, sin considerarse implementación productiva validada.

## Infraestructura, backups y STOP

STORAGE-DATA-PRESERVATION PASS; incidente CLOSED_OPERATIONALLY / MONITORING; **SSD IN SERVICE**. No crear un roadmap de sustitución del SSD; reabrir solo con nueva evidencia objetiva. 238/238 archivos de datos size/SHA256 y 12/12 CSV contra POST PASS; 242 archivos totales sin ausencias/discrepancias. No se reabre como proyecto activo por mantener monitorización.

STOP ante nueva evidencia de I/O error, emergency_ro, SMART warning, USB reset asociado o corrupción. Mantener GitHub para código/docs/artefactos versionables, Drive cifrado para raw/preservación y copia adicional PC/OneDrive. OneDrive no se etiqueta backup auditado independiente sin prueba expresa.

**PROJECT-ARTIFACT-BACKUP-V1** amplía el alcance a artefactos científicos/ML y configuración recuperable, además de raw Forecast. El reconciliador actual no respalda todo el proyecto. Cada nuevo raw prospectivo relevante requiere backup externo. Mantener rclone copy, no sync; validar restore y material crypt recuperable fuera del servidor. El PASS histórico de backup no acredita que una nueva captura ya tenga copia.

## Gobierno y actualización

Los workstreams resuelven problemas operativos dentro de objetivos/invariantes/STOP. Se escalan arquitectura, seguridad, integridad raw y semántica científica. Cambios de estado requieren evidencia y actualización de PROJECT_STATUS; MILESTONES se amplía sin borrar hitos previos. Las fases 2/3 permanecen abiertas y no se declara un RUNNING nuevo sin parte consolidado.


## Ampliación clásica completada — 24/10/2026

✅ HG_RS_V1 con Rs medida incorporado y ejecutado sin calibración;
[comparación oficial de cuatro métodos y CSV](../scientific_pipeline/results/four-classical-20261024T042250Z/README.md), 75 tests/CI PASS.
PM/HS/VAL y Dataset v1 intactos. Próximos gates: PROJECT-ARTIFACT-BACKUP-V1 → AHC-METHODOLOGY-REVIEW → temporal evaluation contract → modelos calibrados simples; esta misión no los implementa.
