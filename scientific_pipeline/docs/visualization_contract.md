commit 7058e640d0ec200a965cac8503b918eca9f5ec7e
Author: Codex <codex@openai.com>
Date:   Mon Nov 9 13:25:51 2026 +0100

    docs: close out ANN v1 modern ET0 results

diff --git a/scientific_pipeline/docs/visualization_contract.md b/scientific_pipeline/docs/visualization_contract.md
index d6e188b..4b153e2 100644
--- a/scientific_pipeline/docs/visualization_contract.md
+++ b/scientific_pipeline/docs/visualization_contract.md
@@ -1,10 +1,10 @@
-> **Cierre posterior — 01/11/2026: ML-V1-CLASSICAL-MODELS PASS**, autorizado por Dirección 00. Dataset v1, cuatro baselines y AHC PASS; ANN moderna NOT STARTED; WEB-V0.3-ML READY. Los apartados previos de preparación/TEST pendiente se conservan como protocolo o historia, no como estado actual. La ejecución final fue **retrospective_recovery_evaluation**: selección y modelos DEVELOPMENT congelados antes de recovery, intento original TEST_ACCESS_UNKNOWN, sin reselección/entrenamiento/refit después de TEST. Las menciones a una evaluación única describen el diseño inicial, no el historial real; no afirmar holdout prístino/prospectivo. [Resultados, interpretación y linaje](../../docs_1.2.6_Modelos%20ML%20cl%C3%A1sicos%20para%20estimaci%C3%B3n%20de%20ET0.md).
+> **Cierre posterior — 01/11/2026: ML-V1-CLASSICAL-MODELS PASS**, autorizado por Dirección 00. Dataset v1, cuatro baselines y AHC PASS; ANN moderna NOT STARTED **(estado histórico al 01/11/2026, superado el 02/11)**; WEB-V0.3-ML READY. Los apartados previos de preparación/TEST pendiente se conservan como protocolo o historia, no como estado actual. La ejecución final fue **retrospective_recovery_evaluation**: selección y modelos DEVELOPMENT congelados antes de recovery, intento original TEST_ACCESS_UNKNOWN, sin reselección/entrenamiento/refit después de TEST. Las menciones a una evaluación única describen el diseño inicial, no el historial real; no afirmar holdout prístino/prospectivo. [Resultados, interpretación y linaje](../../docs_1.2.6_Modelos%20ML%20cl%C3%A1sicos%20para%20estimaci%C3%B3n%20de%20ET0.md).
 
-> **Estado posterior — 31/10/2026:** WEB-V0.2-AHC ya está desplegada (PR #46/#47) y el runtime ML rolling está implementado (PR #48); rolling validation y Linux smoke PASS. La campaña científica oficial permanece pendiente de despliegue/ejecución; ANN NOT STARTED. Las referencias inferiores a «web no construida», ML futuro o merge pendiente conservan el estado previo de su entrega. No cambian la semántica contractual ni resultados congelados. Para el contrato rolling vigente, véase [refinamiento temporal](ml_validation_contract_refinement.md); estado rápido en [PROJECT_STATUS](../../docs/PROJECT_STATUS.md).
+> **Estado posterior — 31/10/2026:** WEB-V0.2-AHC ya está desplegada (PR #46/#47) y el runtime ML rolling está implementado (PR #48); rolling validation y Linux smoke PASS. La campaña científica oficial permanece pendiente de despliegue/ejecución; ANN NOT STARTED **(estado histórico al 31/10/2026, superado el 02/11)**. Las referencias inferiores a «web no construida», ML futuro o merge pendiente conservan el estado previo de su entrega. No cambian la semántica contractual ni resultados congelados. Para el contrato rolling vigente, véase [refinamiento temporal](ml_validation_contract_refinement.md); estado rápido en [PROJECT_STATUS](../../docs/PROJECT_STATUS.md).
 
 # Contrato de datos para visualización científica v1
 
-**Estado documental vigente — 25/10/2026: FROZEN / AVAILABLE TO 06.** AHC temporal ya está PASS. Este contrato describe consumo/experimentos futuros: no ML/ANN entrenado ni web construida. El consumidor utiliza las tablas/schemas del productor y no recalcula ciencia ni inventa agregados no publicados. [Estado maestro](../../docs/PROJECT_STATUS.md).
+**Estado documental vigente — 25/10/2026: FROZEN / AVAILABLE TO 06.** AHC temporal ya está PASS. Este contrato describe consumo/experimentos futuros: no ML/ANN entrenado ni web construida **(estado histórico, superado posteriormente)**. El consumidor utiliza las tablas/schemas del productor y no recalcula ciencia ni inventa agregados no publicados. [Estado maestro](../../docs/PROJECT_STATUS.md).
 
 El productor AHC genera una vista derivada para un futuro consumidor Web; esta misión **no construye la web**. Este contrato describe archivos del productor, sin afirmar que una ejecución concreta haya pasado: comprobar status, manifest y hashes del lote. Dataset, raw, Quality y baselines permanecen congelados. Ver [metodología AHC](ahc_methodology.md) y [contrato ML](ml_experiment_contract.md).
 
