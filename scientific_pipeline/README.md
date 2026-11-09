commit 7058e640d0ec200a965cac8503b918eca9f5ec7e
Author: Codex <codex@openai.com>
Date:   Mon Nov 9 13:25:51 2026 +0100

    docs: close out ANN v1 modern ET0 results

diff --git a/scientific_pipeline/README.md b/scientific_pipeline/README.md
index e3a94e5..d96cf11 100644
--- a/scientific_pipeline/README.md
+++ b/scientific_pipeline/README.md
@@ -1,8 +1,10 @@
-> **Cierre posterior — 01/11/2026: ML-V1-CLASSICAL-MODELS PASS**, autorizado por Dirección 00. Dataset v1, cuatro baselines y AHC PASS; ANN moderna NOT STARTED; WEB-V0.3-ML READY. Los apartados previos de preparación/TEST pendiente se conservan como protocolo o historia, no como estado actual. La ejecución final fue **retrospective_recovery_evaluation**: selección y modelos DEVELOPMENT congelados antes de recovery, intento original TEST_ACCESS_UNKNOWN, sin reselección/entrenamiento/refit después de TEST. Las menciones a una evaluación única describen el diseño inicial, no el historial real; no afirmar holdout prístino/prospectivo. [Resultados, interpretación y linaje](../docs_1.2.6_Modelos%20ML%20cl%C3%A1sicos%20para%20estimaci%C3%B3n%20de%20ET0.md).
+> **Estado posterior — 02/11/2026: ANN-V1-MODERN-ET0 PASS.** La ANN moderna `FEATURE_SET_FULL / MLP [32]` aproxima `PM_FAO56_V1` calculada; no valida ET₀ instrumental independiente. TEST 2021–2025 es `retrospective_locked`, sin selección/entrenamiento/refit, y no globalmente prístino. Resultado y linaje: [documentación ANN](../docs_2.1_Estimación%20Et0%20ANN%20%28tensorflow%29.md). Las referencias inferiores a ANN no iniciada son históricas.
+
+> **Cierre posterior — 01/11/2026: ML-V1-CLASSICAL-MODELS PASS**, autorizado por Dirección 00. Dataset v1, cuatro baselines y AHC PASS; ANN moderna NOT STARTED **(estado histórico al 01/11/2026, superado el 02/11)**; WEB-V0.3-ML READY. Los apartados previos de preparación/TEST pendiente se conservan como protocolo o historia, no como estado actual. La ejecución final fue **retrospective_recovery_evaluation**: selección y modelos DEVELOPMENT congelados antes de recovery, intento original TEST_ACCESS_UNKNOWN, sin reselección/entrenamiento/refit después de TEST. Las menciones a una evaluación única describen el diseño inicial, no el historial real; no afirmar holdout prístino/prospectivo. [Resultados, interpretación y linaje](../docs_1.2.6_Modelos%20ML%20cl%C3%A1sicos%20para%20estimaci%C3%B3n%20de%20ET0.md).
 
 > **Estado posterior de preservación — 01/11/2026:** WINDOWS-SCIENCE-INGEST-AUTOMATION PASS; PROJECT-ARTIFACT-BACKUP-V1 PASS / MAINTAINED. Las referencias inferiores a transferencia manual/Windows sin protección automática describen el cierre anterior. El límite vigente es finalización/preparación y READY firmado explícitos; después, con clave SFTP desbloqueada en ssh-agent, transferencia, verificación servidor, ingestión y preservación automáticas. [Contrato operativo y evidencia E2E](../docs/backup/windows-ingestion.md). Esta nota no modifica ciencia ni operación Forecast.
 
-> **Estado vigente — 31/10/2026:** Dataset v1, cuatro baselines y AHC oficiales PASS. WEB-V0.2-AHC desplegada, sin recalcular ciencia. ML rolling validation (PR #45), Linux smoke y runtime (PR #48) PASS. La campaña clásica se interrumpió después de persistir la selección y el bundle DEVELOPMENT; el acceso a TEST de ese intento es desconocido. Está en curso la herramienta aislada para recuperación de evaluación, sin volver a entrenar ni seleccionar. ANN NOT STARTED. Los apartados fechados 24–25/10 conservan estados anteriores. [Contrato rolling](docs/ml_validation_contract_refinement.md), [estado maestro](../docs/PROJECT_STATUS.md) y [localizaciones/evidencia](../docs/DATA_LOCATIONS.md).
+> **Estado vigente — 31/10/2026:** Dataset v1, cuatro baselines y AHC oficiales PASS. WEB-V0.2-AHC desplegada, sin recalcular ciencia. ML rolling validation (PR #45), Linux smoke y runtime (PR #48) PASS. La campaña clásica se interrumpió después de persistir la selección y el bundle DEVELOPMENT; el acceso a TEST de ese intento es desconocido. Está en curso la herramienta aislada para recuperación de evaluación, sin volver a entrenar ni seleccionar. ANN NOT STARTED **(estado histórico, superado el 02/11)**. Los apartados fechados 24–25/10 conservan estados anteriores. [Contrato rolling](docs/ml_validation_contract_refinement.md), [estado maestro](../docs/PROJECT_STATUS.md) y [localizaciones/evidencia](../docs/DATA_LOCATIONS.md).
 
 # SIAR Science — ET₀ baselines v1
 
@@ -179,7 +181,7 @@ común de cuatro 70860. Dataset v1 y resultados anteriores intactos.
 
 **PROJECT-ARTIFACT-BACKUP-V1 PASS**: snapshot REMOTE_VERIFIED de 128 archivos, incluyendo scope científico y raw SIAR publicado; [DATA_LOCATIONS](../docs/DATA_LOCATIONS.md). La evidencia científica que aún dice «backup pendiente» refleja su publicación anterior al cierre de preservación y se mantiene inmutable. Futuros resultados solo Windows NO están automáticamente protegidos: requieren transferencia/ingestión al servidor. Daily reconciliation general para material ingerido, distinta de la reconciliación SIAR cada 15 min.
 
-Siguiente etapa: ML v1 bajo [contrato FROZEN/READY](docs/ml_experiment_contract.md); [visualization contract FROZEN/AVAILABLE TO 06](docs/visualization_contract.md). No ML/ANN entrenado ni web construida. No se reabren metodología AHC ni baselines ni se recalcula ciencia desde UI.
+Siguiente etapa: ML v1 bajo [contrato FROZEN/READY](docs/ml_experiment_contract.md); [visualization contract FROZEN/AVAILABLE TO 06](docs/visualization_contract.md). No ML/ANN entrenado ni web construida **(estado histórico, superado posteriormente)**. No se reabren metodología AHC ni baselines ni se recalcula ciencia desde UI.
 
 ## Experimento AHC temporal versionado
 
