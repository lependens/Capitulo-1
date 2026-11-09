commit 7058e640d0ec200a965cac8503b918eca9f5ec7e
Author: Codex <codex@openai.com>
Date:   Mon Nov 9 13:25:51 2026 +0100

    docs: close out ANN v1 modern ET0 results

diff --git a/scientific_pipeline/docs/mission_status.md b/scientific_pipeline/docs/mission_status.md
index 2b39204..de39421 100644
--- a/scientific_pipeline/docs/mission_status.md
+++ b/scientific_pipeline/docs/mission_status.md
@@ -1,8 +1,10 @@
-> **Cierre posterior — 01/11/2026: ML-V1-CLASSICAL-MODELS PASS**, autorizado por Dirección 00. Dataset v1, cuatro baselines y AHC PASS; ANN moderna NOT STARTED; WEB-V0.3-ML READY. Los apartados previos de preparación/TEST pendiente se conservan como protocolo o historia, no como estado actual. La ejecución final fue **retrospective_recovery_evaluation**: selección y modelos DEVELOPMENT congelados antes de recovery, intento original TEST_ACCESS_UNKNOWN, sin reselección/entrenamiento/refit después de TEST. Las menciones a una evaluación única describen el diseño inicial, no el historial real; no afirmar holdout prístino/prospectivo. [Resultados, interpretación y linaje](../../docs_1.2.6_Modelos%20ML%20cl%C3%A1sicos%20para%20estimaci%C3%B3n%20de%20ET0.md).
+> **Estado posterior — 02/11/2026: ANN-V1-MODERN-ET0 PASS.** La ANN moderna `FEATURE_SET_FULL / MLP [32]` aproxima `PM_FAO56_V1` calculada; no valida ET₀ instrumental independiente. TEST 2021–2025 es `retrospective_locked`, sin selección/entrenamiento/refit, y no globalmente prístino. Resultado y linaje: [documentación ANN](../../docs_2.1_Estimación%20Et0%20ANN%20%28tensorflow%29.md). Las referencias inferiores a ANN no iniciada son históricas.
+
+> **Cierre posterior — 01/11/2026: ML-V1-CLASSICAL-MODELS PASS**, autorizado por Dirección 00. Dataset v1, cuatro baselines y AHC PASS; ANN moderna NOT STARTED **(estado histórico al 01/11/2026, superado el 02/11)**; WEB-V0.3-ML READY. Los apartados previos de preparación/TEST pendiente se conservan como protocolo o historia, no como estado actual. La ejecución final fue **retrospective_recovery_evaluation**: selección y modelos DEVELOPMENT congelados antes de recovery, intento original TEST_ACCESS_UNKNOWN, sin reselección/entrenamiento/refit después de TEST. Las menciones a una evaluación única describen el diseño inicial, no el historial real; no afirmar holdout prístino/prospectivo. [Resultados, interpretación y linaje](../../docs_1.2.6_Modelos%20ML%20cl%C3%A1sicos%20para%20estimaci%C3%B3n%20de%20ET0.md).
 
 > **Estado posterior de preservación — 01/11/2026:** WINDOWS-SCIENCE-INGEST-AUTOMATION PASS; PROJECT-ARTIFACT-BACKUP-V1 PASS / MAINTAINED. Las referencias inferiores a transferencia manual/Windows sin protección automática describen el cierre anterior. El límite vigente es finalización/preparación y READY firmado explícitos; después, con clave SFTP desbloqueada en ssh-agent, transferencia, verificación servidor, ingestión y preservación automáticas. [Contrato operativo y evidencia E2E](../../docs/backup/windows-ingestion.md). Esta nota no modifica ciencia ni operación Forecast.
 
-> **Estado posterior — 31/10/2026:** WEB-V0.2-AHC ya está desplegada (PR #46/#47) y el runtime ML rolling está implementado (PR #48); rolling validation y Linux smoke PASS. La campaña científica oficial permanece pendiente de despliegue/ejecución; ANN NOT STARTED. Las referencias inferiores a «web no construida», ML futuro o merge pendiente conservan el estado previo de su entrega. No cambian la semántica contractual ni resultados congelados. Para el contrato rolling vigente, véase [refinamiento temporal](ml_validation_contract_refinement.md); estado rápido en [PROJECT_STATUS](../../docs/PROJECT_STATUS.md).
+> **Estado posterior — 31/10/2026:** WEB-V0.2-AHC ya está desplegada (PR #46/#47) y el runtime ML rolling está implementado (PR #48); rolling validation y Linux smoke PASS. La campaña científica oficial permanece pendiente de despliegue/ejecución; ANN NOT STARTED **(estado histórico al 31/10/2026, superado el 02/11)**. Las referencias inferiores a «web no construida», ML futuro o merge pendiente conservan el estado previo de su entrega. No cambian la semántica contractual ni resultados congelados. Para el contrato rolling vigente, véase [refinamiento temporal](ml_validation_contract_refinement.md); estado rápido en [PROJECT_STATUS](../../docs/PROJECT_STATUS.md).
 
 # ET0-BASELINES-PREP — estado de integración
 
@@ -14,7 +16,7 @@ AHC_CLIMATOLOGY se calcula después de TEST con todo el histórico declarado has
 
 **PROJECT-ARTIFACT-BACKUP-V1 PASS** según cierre de 02 comunicado por Dirección 00; snapshot backup_20261024T215639Z_63d0ef01 REMOTE_VERIFIED, 128 archivos. Los artefactos científicos cubiertos ya tienen preservación externa; no implica que cualquier resultado futuro solo Windows esté automáticamente protegido. Requiere transferencia/ingestión explícita y reconciliación diaria para lo ingerido. [Alcance, pruebas y límites](../../docs/DATA_LOCATIONS.md).
 
-ML contract FROZEN/READY FOR FUTURE EXPERIMENTS y visualization contract FROZEN/AVAILABLE TO 06; no ML/ANN entrenado ni web construida.
+ML contract FROZEN/READY FOR FUTURE EXPERIMENTS y visualization contract FROZEN/AVAILABLE TO 06; no ML/ANN entrenado ni web construida **(estado histórico, superado posteriormente)**.
 
 
 ## Registro de cuatro métodos — 24/10/2026
