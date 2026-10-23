# Auditoría documental — DOCUMENTATION-TRUTH-REFRESH-2026-10

**Fecha: 2026-10-23, Europe/Madrid.** Base revisada: `76d531cd3dfe675bf1f2dac2c81eab1028e1c11e`. El encargo partía de `2c4874c...`; antes de editar ya estaban fusionadas PR #30/#31. Se adoptó esa evidencia posterior, sin inferir un estado nuevo de timers.

## Fuentes y jerarquía

- Dirección 00: estado operativo de Sync HOLD, timers pausados, storage cerrado/monitorización y dependencia Forecast desacoplada.
- POST corregido, revisión independiente y corrección del ledger: evidencia Quality vigente; snapshots originales inmutables.
- Run oficial PR #31, result-summary.json y baselines/manifest.json SUCCESS: dataset, recuentos y resultados PM/HS vigentes.
- PROJECT_STATUS: estado rápido; ROADMAP: futuro/gates; MILESTONES: historia append-only.
- Bitácoras: metodología histórica preservada más actualización vigente. Componentes: contrato técnico.
- Esta misión no consulta producción, lanza timers, ejecuta baselines, modifica raw ni genera artefactos científicos.

## Matriz de inconsistencias y disposición

| Documento o grupo revisado | Hallazgo | Corrección / clasificación |
|---|---|---|
| README.md | POST sin artefactos, dataset futuro, R2 ejecutándose y dependencias Forecast antiguas | Mapa, resumen y próximos pasos vigentes; enlace a ubicaciones/resultados |
| docs/PROJECT_STATUS.md | Estado maestro anterior al cierre Quality y a resultados oficiales | Consolidación vigente y compacta; capacidades técnicas detalladas siguen en sus componentes/bitácoras |
| docs/ROADMAP.md | R2/POST/dataset como próximos hitos; dependencia PILOT para AEMET | Gates satisfechos identificados; Sync HOLD separado; AEMET tras CAPTURE; Valiantzas antes de ML |
| docs/MILESTONES.md | Faltan hitos de cierre y ejecución oficial | Solo nuevas entradas; contenido anterior intacto |
| docs_1.0 | Objetivos ya alcanzados presentados sin contexto posterior | Aviso histórico y actualización final |
| docs_1.1 | Recovery activo y POST/dataset pendientes; piloto en observación anterior al incidente | Historia íntegra + actualización del cierre, HOLD y pausas/gates actuales |
| docs_1.2 | Dataset/ejecución pendientes | Historia íntegra + dataset/run PASS, fix VPD y Valiantzas LEGACY / FUTURE BASELINE REVIEW |
| docs_1.3 | Resultados legacy podían confundirse con v1 | Se conserva análisis/tablas; solo aviso y enlace al informe oficial PASS, sin nuevas tablas |
| docs_1.4 y docs_2.1 | ANN histórica sin delimitar su relación con v1 | Historia preservada; ML futuro con split temporal y baselines clásicos previos |
| docs/siar/quality.md | Recovery/POST descritos en curso | Se corrige contexto de ejecución; políticas y encabezados CSV conservados |
| docs/siar/sync-status.md | Mezcla POST original/vigente y handoff futuro | POST corregido, recuentos, dual-SHA, dataset/run actuales y Sync HOLD |
| data_quality/siar/README.md | Resumen original podía confundirse con ledger vigente | Prefacio vigente; original etiquetado superseded |
| siar-sync/README.md | Producción activa y GitHub pendiente de una PR ya fusionada | v2.3.3 canónica, HOLD, alignment e incremental 2026 pendientes |
| forecast_collector/README.md | Scheduler nunca activo como estado actual | Pausa reportada, gates de reactivación y target funcional; contrato preservado |
| aemet_forecast_collector/README.md | Handoff tras PILOT y despliegue aún enteramente futuro | Reactivación tras CAPTURE, independiente de Sync; no se inventa nuevo RUNNING |
| scientific_pipeline/README.md | POST/dataset/ejecución todavía pendientes | Estado oficial PASS y separación evidencia compacta / archivos completos |
| scientific_pipeline/docs/mission_status.md | Preparación histórica presentada como vigente | Nuevo estado vigente; registro de preparación explícitamente histórico |
| scientific_pipeline/docs/contracts.md | Nota inicial de ausencia de dataset ya superada | Contexto actualizado; contrato y semántica científica conservados |
| scientific_pipeline/docs/legacy_audit.md | Espera de POST dentro del informe de investigación | Prefacio que delimita historia y enlaza resultado posterior; cifras intactas |
| scientific_pipeline/docs/units_offline_evidence.md y units_offline_forensic.md | Diagnósticos fechados podían confundirse con ejecución oficial | Aviso histórico; evidencia/cifras preservadas |
| scientific_pipeline/docs/assumptions.md | Ya incorpora tolerancia VPD y metodología vigente | Revisado, sin cambios |
| data_quality/siar/ledger-correction.md | Evidencia de causa/fix | Revisado, sin cambios |
| POST original y tres informes Markdown del POST corregido | Evidencia histórica y certificación | Revisados, sin cambios; no se edita snapshot |
| scientific_pipeline/results/pm-vpd-20261023T194553Z/README.md | PASS y métricas oficiales posteriores al encargo | Revisado como autoridad, sin cambios |
| docs/DATA_LOCATIONS.md | Faltaba guía central de artefactos/localizaciones | Nueva guía; scientific-results/observational-v1 es propuesta, no estructura desplegada |

## Validación y límites

29 documentos preexistentes revisados; 22 modificados y 7 conservados sin cambios. Dos documentos nuevos: esta matriz y DATA_LOCATIONS. Total del cambio: **24 documentos Markdown**.

Revisión de enlaces internos contra el árbol Git y coherencia cruzada; git diff --check y conservación de modos de archivo. No se detectaron enlaces internos rotos preexistentes entre los documentos auditados; no se atribuyen arreglos inexistentes. Los enlaces nuevos deben resolver antes del merge. No se acredita disponibilidad de endpoints externos ni de rutas locales del servidor/PC mediante un enlace.

Las seis bitácoras conservan todo el texto anterior. MILESTONES conserva su prefijo exacto. PROJECT_STATUS y ROADMAP se consolidan para eliminar duplicación de contratos y estados obsoletos; la metodología y las evidencias fechadas permanecen en las bitácoras, componentes, hitos y su historial Git.

Limitaciones documentales restantes: estado de timers según último parte recibido, sin nuevo gate automático demostrado; JSON dataset y predicciones completos fuera de Git con paths del productor; acceso actual a esos discos no probado por esta misión; recuperabilidad externa de material crypt y cobertura de nuevos raw requieren sus propias evidencias. OneDrive no se declara backup auditado independiente.

No se confunde observación ausente confirmada con auditoría incompleta ni métrica contra EtPMon con exactitud instrumental. No se realizan cambios científicos para resolver redacción.
