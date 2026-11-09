commit 7058e640d0ec200a965cac8503b918eca9f5ec7e
Author: Codex <codex@openai.com>
Date:   Mon Nov 9 13:25:51 2026 +0100

    docs: close out ANN v1 modern ET0 results

diff --git a/scientific_pipeline/docs/ml_v1_classical_methodology.md b/scientific_pipeline/docs/ml_v1_classical_methodology.md
index 35a12d4..c8261b3 100644
--- a/scientific_pipeline/docs/ml_v1_classical_methodology.md
+++ b/scientific_pipeline/docs/ml_v1_classical_methodology.md
@@ -1,4 +1,6 @@
-> **Cierre posterior — 01/11/2026: ML-V1-CLASSICAL-MODELS PASS**, autorizado por Dirección 00. Dataset v1, cuatro baselines y AHC PASS; ANN moderna NOT STARTED; WEB-V0.3-ML READY. Los apartados previos de preparación/TEST pendiente se conservan como protocolo o historia, no como estado actual. La ejecución final fue **retrospective_recovery_evaluation**: selección y modelos DEVELOPMENT congelados antes de recovery, intento original TEST_ACCESS_UNKNOWN, sin reselección/entrenamiento/refit después de TEST. Las menciones a una evaluación única describen el diseño inicial, no el historial real; no afirmar holdout prístino/prospectivo. [Resultados, interpretación y linaje](../../docs_1.2.6_Modelos%20ML%20cl%C3%A1sicos%20para%20estimaci%C3%B3n%20de%20ET0.md).
+> **Estado posterior — 02/11/2026: ANN-V1-MODERN-ET0 PASS.** La ANN moderna `FEATURE_SET_FULL / MLP [32]` aproxima `PM_FAO56_V1` calculada; no valida ET₀ instrumental independiente. TEST 2021–2025 es `retrospective_locked`, sin selección/entrenamiento/refit, y no globalmente prístino. Resultado y linaje: [documentación ANN](../../docs_2.1_Estimación%20Et0%20ANN%20%28tensorflow%29.md). Las referencias inferiores a ANN no iniciada son históricas.
+
+> **Cierre posterior — 01/11/2026: ML-V1-CLASSICAL-MODELS PASS**, autorizado por Dirección 00. Dataset v1, cuatro baselines y AHC PASS; ANN moderna NOT STARTED **(estado histórico al 01/11/2026, superado el 02/11)**; WEB-V0.3-ML READY. Los apartados previos de preparación/TEST pendiente se conservan como protocolo o historia, no como estado actual. La ejecución final fue **retrospective_recovery_evaluation**: selección y modelos DEVELOPMENT congelados antes de recovery, intento original TEST_ACCESS_UNKNOWN, sin reselección/entrenamiento/refit después de TEST. Las menciones a una evaluación única describen el diseño inicial, no el historial real; no afirmar holdout prístino/prospectivo. [Resultados, interpretación y linaje](../../docs_1.2.6_Modelos%20ML%20cl%C3%A1sicos%20para%20estimaci%C3%B3n%20de%20ET0.md).
 
 # ML v1 — campaña clásica bloqueada antes del entrenamiento científico
 
