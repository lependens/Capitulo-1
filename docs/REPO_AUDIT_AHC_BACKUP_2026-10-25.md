# REPO-TRUTH-REFRESH-AHC-BACKUP-2026-10-24 — consolidación

**Fecha documental: 2026-10-25 (Europe/Madrid).** Base verificada d15bbffac99164994aa884c1a47dc680d54e66af; cero PR abiertas al inicio. La misión conserva el identificador 2026-10-24 de Dirección 00.

## Fuentes y conciliación temporal

42 documentos Markdown preexistentes revisados, incluidos documentación maestra/bitácoras/componentes, todos los contratos científicos y README de resultados; ocho JSON de protocolo/evidencia AHC leídos. PR #38/#39 merged. Los manifests TRAIN/VALIDATION/TEST/closeout y view-manifest son SUCCESS; verification.json PASS, cifras comunes contra PM e identidades coinciden con el encargo.

La evidencia AHC se publicó antes del snapshot backup_20261024T215639Z_63d0ef01. Su README/artifact-index todavía registran backup no certificado en ese momento: **son evidencia fechada preservada**, no contradicción con el cierre posterior de 02. No se editan esos archivos ni sus checksums. El estado de preservación vigente procede del parte de cierre de 02 comunicado por Dirección 00, no de una nueva inspección de servidor/Drive.

## Matriz de estados superados

| Ámbito | Estado superado | Corrección vigente |
|---|---|---|
| PROJECT_STATUS / ROADMAP | AHC y PROJECT-ARTIFACT-BACKUP-V1 pendientes | Ambos PASS; ML v1 siguiente y contratos congelados |
| DATA_LOCATIONS | Dataset/CSV masivos solo locales, sin preservación externa demostrada | Snapshot REMOTE_VERIFIED de 128 archivos; hashes dataset/CSV PASS; paths de origen y copia remota distinguidos |
| README | Preservación pendiente y revisión AHC como próximo gate | Cierres resumidos; navegación al estado/contratos/evidencia |
| Bitácoras docs_1.x/docs_2.x | Planes y snapshots anteriores | Texto previo intacto y actualización final fechada; AHC legacy vs validado vs climatología |
| scientific_pipeline README / assumptions / contracts / mission_status | Metodología no decidida y backup pendiente | OLS0 TRAIN y split cerrado; preservación con límites, sin cambiar código/fórmulas de baselines |
| AHC handoff 05→03 | Backup pendiente a la fecha de publicación | Nota posterior explícita, contenido científico previo conservado |
| ML experiment contract | Título propuesta y gate AHC anterior | FROZEN/READY FOR FUTURE EXPERIMENTS, sin modelos entrenados; protocolo de ejecución sigue necesario |
| Visualization contract | Contrato sin estado de disponibilidad explícito | FROZEN/AVAILABLE TO 06, sin web ni recálculo de ciencia |
| Forecast README | Política global de preservación por completar | Backup general PASS con scope; gate de nuevo run automático no cambia |
| MILESTONES | Gates aún pendientes en entradas históricas | Solo nuevas entradas de cierre, prefijo original exacto |
| Quality/Sync/AEMET/resultados e índices versionados | Estados o evidencia ya correctos | Revisados; no cambios raw/Quality/Dataset, no servicios ni nuevos raws AEMET |

## Invariantes y límites

- AHC_VALIDATED OLS0 por estación, PM primaria, EtPMon secundaria calculada; no ET medida.
- TRAIN 2004–2015, VALIDATION 2016–2020, TEST 2021–2025 retrospective_locked; sin refit ni reselección.
- HG-Rs seasonal elegido en VALIDATION, no una selección posterior TEST. Climatología después de TEST y separada; estabilidad del coeficiente no garantiza estabilidad del error.
- 128 verificados = 82 scientific-results (1,421,919,603 bytes) + 46 SIAR published Raw. Inventario 221 no equivale a backup 221.
- cryptcheck y restores 2 científicos/2 referencias operativas con hashes PASS; descarga redundante completa INTERRUPTED, nunca full-download verification PASS.
- Ingestión event-driven PASS (new=0, unchanged=9) y daily reconciliation ENABLED solo para material ya ingerido. SIAR publicado mantiene reconciliación propia cada 15 min.
- Futuros artefactos solo Windows NOT AUTOMATICALLY PROTECTED: necesitan transferencia/ingestión al servidor.
- Config secreta/credenciales excluidas; material crypt/OAuth OFF-SERVER sin publicarlo. Closeout se referencia por path, no se copia.
- Forecast unchanged: captura/reconciliador ACTIVE, nuevo run real automático + backup + COMPLETE remoto pendiente. AEMET disabled/inactive, captures=0, gate CLOSED, espera CAPTURE RUNNING. Sync HOLD.

Validación: alcance Markdown, diff-check, enlaces internos, preservación de historia/MILESTONES, modos y escaneo del diff. No nueva auditoría raw, ejecución científica, configuración ni operación de infraestructura. Las auditorías anteriores permanecen históricas e intactas; no se actualizan sus conteos retrospectivamente.
