# ROADMAP — SIAR / ET₀

**Actualizado: 2026-10-24 (Europe/Madrid).** Próximos gates y dependencias; estado demostrado en [PROJECT_STATUS](PROJECT_STATUS.md), hitos en [MILESTONES](MILESTONES.md) y ubicaciones en [DATA_LOCATIONS](DATA_LOCATIONS.md).

## Leyenda y trabajo paralelo

✅ demostrado; 🟠 pendiente de validación o investigación; ⚪ futuro. Histórico y superado se fechan explícitamente. Un resultado Quality COMPLETE no implica observaciones para el 100 % de días.

| Línea | Ya demostrado | Siguiente gate |
|---|---|---|
| Observaciones / Quality | POST corregido COMPLETE; 2.345 ausencias confirmadas, 0 PENDING_RETRY | Mantener snapshot cerrado; CATALOG-TZ-001 no bloqueante |
| Ciencia | Dataset v1 y comparación oficial PM/HS/VAL/HG-Rs PASS | Preservación y contrato temporal antes de calibración/ML |
| SIAR Sync / 02 | v2.3.3 canónica, snapshot científico 2025 cerrado | HOLD → production alignment + incremental 2026 demostrado |
| Forecast SIAR / 02 | Runtime, raw, backup/restore; 8 estaciones funcionales | Reactivar timers pausados → 1 ciclo real CAPTURE RUNNING → mínimo 2 PILOT RUNNING |
| AEMET Track A / 02 | Collector y E2E DEMONSTRATED; piloto pausado | Tras SIAR-FORECAST-CAPTURE RUNNING, reactivar prospectivo |
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

Último parte operativo: timers pausados tras incidente. Próximos gates:

1. 🟠 Reactivación prospectiva independiente de SIAR Sync.
2. 🟠 Un ciclo automático real con evidencia → **SIAR-FORECAST-CAPTURE RUNNING**.
3. 🟠 Al menos dos ciclos automáticos → **SIAR-FORECAST-PILOT RUNNING**.
4. 🟠 Asegurar backup externo de cada nuevo raw relevante y recuperabilidad del material crypt.
5. 🟠 Acumular observación operacional y comprobar cobertura/cadencia, sin asignar D+n prematuramente.

Target funcional: IB01, IB02, IB03, IB04, IB05, IB06, IB10, IB11. IB08 sigue ACTIVE / TEMPORARILY_UNAVAILABLE_EXTERNAL, vigilada aparte; no bloquea las ocho ni se etiqueta CLOSED. Cadencia experimental 08:00 / 14:00 / 20:00 Europe/Madrid.

source issue time puede ser null; D+0…D+6 y recuperación histórica de emisiones no están demostrados. No rellenar la pausa retrospectivamente. Salida de fase: captura y backup sostenidos con evidencia; un PASS de smoke no equivale al gate periódico.

## Fase 3 — AEMET

✅ Track A municipal diario OpenData: E2E DEMONSTRATED, estabilización PASS. PR #18/#19/#20: collector, id municipal y permisos POSIX. Conserva wrapper.raw y payload.raw sin modificar bytes; id 7040 se compara canónicamente con 07040 sin editar raw.

Piloto IB04 → Palma → municipio 07040, capital municipal, no punto exacto de estación. Último parte: pausado tras incidente.

**Dependencia vigente:** SIAR-FORECAST-CAPTURE RUNNING → reactivar AEMET Track A. Se supera la dependencia antigua de esperar SIAR-FORECAST-PILOT RUNNING; nunca depende de SIAR Sync.

🟠 Validar reactivación, runtime/configuración, raw independiente /srv/aemet-forecast/raw, backup rclone crypt, restore y scheduler experimental con evidencia de los nuevos ciclos. No se infiere ejecución actual del mero contrato del collector.

ET₀ NO; D+n NO; timezone de emisión inferida NO; integración científica SIAR/AEMET NO; histórico de emisiones no demostrado. El producto no aporta por sí solo entradas suficientes para FAO-56 PM completo.

🟠 **Track B OPEN:** investigar feed AEMET exacto utilizado por SIAR, independiente de Track A.

## Fase 4 — Dataset Observational v1

✅ Construido y validado estructuralmente con POST corregido y handoff autorizado. Dataset siar-balears-observational-v1; SHA256 **0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb**.

Conserva 75.496 observaciones raw, 23 columnas, una fila exterior, ledger y provenance; no interpola, imputa ni fabrica meteorología. El JSON completo es un artefacto local preservado, no se afirma almacenado íntegro en Git. Referencias verificables en [ubicaciones](DATA_LOCATIONS.md).

🟠 Mantener preservación, identidad y acceso; revisar convención de resultados con el productor antes de mover archivos. scientific-results/observational-v1/{dataset,baselines,exports} es solo una propuesta, no un despliegue ni un movimiento aprobado.

## Fase 5 — Baselines clásicos y comparación

✅ PM_FAO56_V1 y HS_RA_V1 implementados y primera ejecución oficial PASS, publicada en PR #31 tras fix numérico PR #30. 62 tests PASS y CI 3.11/3.12/3.13 según evidencia publicada. Dataset y Quality intactos durante el fix.

PM y HS usan Tcalc=(Tmax+Tmin)/2. PM: G=0, gamma=0.000665 P, albedo 0.23, viento m/s bajo convención SIAR 2 m, radiación MJ/m²/día. HS: 0.0023 × 0.408 × Ra × (Tcalc+17.8) × sqrt(Tmax−Tmin), sin radiación medida ni calibración.

Los resultados oficiales, cohortes y limitaciones están en [la ejecución publicada](../scientific_pipeline/results/pm-vpd-20261023T194553Z/README.md). No comparar métricas legacy con v1 como si compartieran máscara, periodo o metodología. EtPMon es calculada SIAR, no observación independiente.

Próxima secuencia:

1. ✅ Fórmula y unidades auditadas; Martí et al. (2024), composición Ecs.1b+3 con conversión 0.408 aprobada.
2. ✅ **VALIANTZAS_V1** implementado, 68 tests y CI PASS; PR #33.
3. ✅ [Comparación oficial PM / HS / VAL y exportación CSV](../scientific_pipeline/results/classical-20261024T034529Z/README.md): cohortes propias, pares y triple, global/estación/año/mes/temporada.
4. 🟠 Retener cobertura/limitaciones, especialmente IB09 excluida por FAIL y muestra corta IB101.
5. Después avanzar a nuevos modelos.

No copiar resultados antiguos ni calibrar constantes para reducir artificialmente el error SIAR. UNITS_PROVENANCE_PARTIAL y procedencia instrumental no exhaustiva siguen como limitaciones, no STOP global.

## Fase 6 — Modelos estadísticos y ML

⚪ Regresión y estadística simple → árboles → boosting → ANN más adelante. Usar partición temporal global TRAIN < VALIDATION < TEST, sin leakage. Transformaciones, filtros aprendidos, escalas y cualquier AHC se ajustan exclusivamente en TRAIN. Definir particiones y objetivos antes de ajustar; no reutilizar la calibración in-sample histórica como validación.

Salida: comparación reproducible frente a baselines clásicos con métricas n/MAE/RMSE/bias/R², cobertura y provenance, no solo una mejora agregada sin cohorte.

## Fase 7 — Evaluación predictiva

⚪ Acumular raw SIAR y AEMET prospectivamente → resolver issue time / valid time / D+n y correspondencia estación/municipio → datasets forecast derivados → comparar con observaciones posteriores.

No asignar horizonte por posición de fila ni timezone por conjetura. No mezclar forecast municipal con estación puntual sin explicitar correspondencia y límites. No esperar a terminar los modelos para comenzar la acumulación prospectiva.

## Fase 8 — Producto

⚪ Web/API → ET₀ + Kc → ETc → necesidades de riego. Separar adquisición, procesamiento científico, modelado y producto. Los prototipos ANN y dashboards legacy conservan valor histórico, sin considerarse implementación productiva validada.

## Infraestructura, backups y STOP

STORAGE-DATA-PRESERVATION PASS; Toshiba CLOSED_OPERATIONALLY / MONITORING. 238/238 archivos de datos size/SHA256 y 12/12 CSV contra POST PASS; 242 archivos totales sin ausencias/discrepancias. No se reabre como proyecto activo por mantener monitorización.

STOP ante nueva evidencia de I/O error, emergency_ro, SMART warning, USB reset asociado o corrupción. Mantener GitHub para código/docs/artefactos versionables, Drive cifrado para raw/preservación y copia adicional PC/OneDrive. OneDrive no se etiqueta backup auditado independiente sin prueba expresa.

Cada nuevo raw prospectivo relevante requiere backup externo. Mantener rclone copy, no sync; validar restore y material crypt recuperable fuera del servidor. El PASS histórico de backup no acredita que una nueva captura ya tenga copia.

## Gobierno y actualización

Los workstreams resuelven problemas operativos dentro de objetivos/invariantes/STOP. Se escalan arquitectura, seguridad, integridad raw y semántica científica. Cambios de estado requieren evidencia y actualización de PROJECT_STATUS; MILESTONES se amplía sin borrar hitos previos. Las fases 2/3 permanecen abiertas y no se declara un RUNNING nuevo sin parte consolidado.


## Ampliación clásica completada — 24/10/2026

✅ HG_RS_V1 con Rs medida incorporado y ejecutado sin calibración;
[comparación oficial de cuatro métodos y CSV](../scientific_pipeline/results/four-classical-20261024T042250Z/README.md), 75 tests/CI PASS.
PM/HS/VAL y Dataset v1 intactos. Próximo gate científico: definir objetivo y
partición temporal antes de cualquier ajuste o ML; esta misión no los inicia.
