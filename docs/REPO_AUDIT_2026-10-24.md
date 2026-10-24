# REPO-TRUTH-REFRESH-2026-10-24 — auditoría documental

Base contrastada: **9f3dde8e22ab8e8f76aeaa273b5e7681558ebbcf**; sin PR abiertas al inicio. Se releen las fuentes en main y el parte operativo de Dirección 00 del 24/10. No se accede a producción, a la config rclone ni al workspace Windows del productor.

## Inventario de revisión y evidencia

**36 documentos Markdown preexistentes** revisados: README; PROJECT_STATUS/ROADMAP/MILESTONES/DATA_LOCATIONS y auditoría anterior; seis bitácoras docs_1.x/docs_2.x; docs/siar; data_quality/siar README, corrección y cuatro informes de snapshots; README de Sync, SIAR Forecast, AEMET y SECURITY_EVIDENCE; README científico, ocho documentos científicos y tres README de resultados.

Además se contrastaron **13 índices/manifiestos/resúmenes JSON versionados**, incluidos los tres runs científicos y el experimento forense. El resumen y manifest de cuatro confirman SUCCESS/PASS, dataset intacto, HG-Rs 70.978 calculadas / 70.912 evaluables / 0 fallos, común 70.860 y el CSV externo con SHA256 f55e4ed12691b84dbca4e090c49da538d5e7fad4f2ef371e1307a649cfb1af44. No se vuelve a calcular ni se certifica acceso actual a esos archivos externos.

Fuentes canónicas: [VAL v1](../scientific_pipeline/docs/valiantzas_v1.md), [HG-Rs v1](../scientific_pipeline/docs/hargreaves_rs_v1.md), [resultados de cuatro](../scientific_pipeline/results/four-classical-20261024T042250Z/README.md), PR #33–#36 y CI 40062004170 success. Los resultados/índices/manifiestos y artefactos Quality quedan inmutables.

## Matriz de inconsistencias

| Ámbito | Antes | Disposición vigente |
|---|---|---|
| PROJECT_STATUS | Añadidos de tres/cuatro con resumen PM/HS aún dominante; timers pausados | Consolidar cuatro, tabla común, precisión vs inputs, timer ACTIVE, reconciliador y gates pendientes |
| ROADMAP | Valiantzas ya implementado pero ruta inmediata poco consolidada | Baselines PASS → preservación → AHC review → contrato temporal → modelos simples → regresión/árboles/boosting → ANN |
| DATA_LOCATIONS | Tabla inicial enlaza como actuales solo PM/HS aunque exportes ampliados están al final | Tabla de cuatro actual; dataset/productores separados; CSV completo externo visible; gap de backup explícito |
| README | Valiantzas pendiente y timers pausados como estado rápido | Cuatro oficiales PASS; operación activa distinta de CAPTURE/PILOT; preservación y AHC pendientes |
| docs_1.0 / 1.1 / 1.2 / 1.3 / 1.4 / 2.1 | Estados fechados superados en el relato | Aviso y actualización final; todo el texto anterior se conserva, incluidas tablas oficiales y legacy identificadas |
| Forecast SIAR README | Último parte de pausa | Timer ACTIVE; OAuth Production; reconciliador 15 min con alcance limitado; ciclo real con backup/COMPLETE pendiente |
| AEMET README | Pausa genérica, sin mecanismo preciso de backup | Capture disabled/inactive; backup tras intento success/failure, sin timer separado; nuevo backup automático NOT ACTIVE |
| Scientific README / assumptions / contracts | AHC futuro tratado como método ya decidido exclusivamente TRAIN | Revisión separada de usos climatológicos/predictivos; sin elegir todo el dataset vs TRAIN, sin cambiar baselines |
| MILESTONES | Registro previo al cierre de ampliaciones y parte nuevo | Solo nuevas entradas; prefijo previo exacto |
| Scientific mission_status / VAL / HG-Rs docs / resultados | Ya contienen cuatro métodos y decisiones detalladas vigentes | Revisados; mission_status y contratos nuevos mantienen su autoridad técnica; no se altera evidencia |
| docs/siar y data_quality | Quality cerrado y POST corregido ya reflejados | Revisados, sin reiniciar Recovery ni modificar snapshots |
| SIAR Sync README | HOLD correctamente reflejado | Revisado, sin cambios: alignment/incremental 2026 pendientes |
| Storage | Cierre operativo previo y preservación PASS | Mantener cierre, añadir SSD IN SERVICE; no inventar sustitución ni incidencia activa |

## Límites y validación

Cambios exclusivamente Markdown. Se preservan modos de archivo, las seis bitácoras y el prefijo de MILESTONES; no se modifican código, raw, Dataset v1, Quality, legacy, JSON/CSV/manifests ni configuración. Las afirmaciones históricas stale permanecen dentro de su contexto fechado; se corrigen las que se presentan como estado actual.

La política AHC no se cierra ni se implementa. Si AHC se estima de datos es parámetro calibrado; la misión específica determinará usos, datos de ajuste y validación temporal. El in-sample legacy no se convierte en validación predictiva.

Validaciones: enlaces internos contra el árbol Git, coherencia de identidades/métricas con el resumen y manifest oficiales, diff-check, modos, preservación y escaneo del diff añadido. No hay lint Markdown dedicado configurado; CI científica y Forecast se comprueban antes de merge.

Limitaciones restantes: **nuevo run automático real + backup + COMPLETE remoto pendiente**; **PROJECT-ARTIFACT-BACKUP-V1 pendiente**, sin garantía de copia externa de todos los artefactos masivos; AHC y contrato temporal no decididos. El parte de runtime de Dirección 00 es fuente operativa; un timer activo no acredita un ciclo completo ni permite inferir otro estado. No se publica config ni secretos; su SHA reportado no equivale a inspección del contenido.

La auditoría de 23/10 [DOCUMENTATION_AUDIT_2026-10](DOCUMENTATION_AUDIT_2026-10.md) permanece histórica e intacta. Este informe no reescribe sus conteos ni convierte sus conclusiones fechadas en el estado del 24/10.
