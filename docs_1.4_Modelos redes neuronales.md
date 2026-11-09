commit 7058e640d0ec200a965cac8503b918eca9f5ec7e
Author: Codex <codex@openai.com>
Date:   Mon Nov 9 13:25:51 2026 +0100

    docs: close out ANN v1 modern ET0 results

diff --git a/docs_1.4_Modelos redes neuronales.md b/docs_1.4_Modelos redes neuronales.md
index a3caa7a..8c62ca6 100755
--- a/docs_1.4_Modelos redes neuronales.md	
+++ b/docs_1.4_Modelos redes neuronales.md	
@@ -167,3 +167,9 @@ AHC temporal y PROJECT-ARTIFACT-BACKUP-V1 PASS superan los gates pendientes ante
 AHC_VALIDATED es OLS0 por estación contra PM, solo TRAIN 2004–2015; VALIDATION 2016–2020 selecciona y TEST 2021–2025 retrospective_locked evalúa sin refit. AHC_CLIMATOLOGY posterior a TEST permanece separado. Los modelos ANN antiguos siguen legacy y no reciben resultados de esta validación.
 
 [Visualization contract](scientific_pipeline/docs/visualization_contract.md) FROZEN/AVAILABLE TO 06, sin web construida ni recálculo de ciencia desde UI. Los artefactos Windows futuros requieren ingestión explícita para entrar en backup; no están automáticamente protegidos por el snapshot existente.
+
+---
+
+## ACTUALIZACIÓN 02/11/2026 — ANN moderna v1
+
+Los resultados de este documento siguen siendo **HISTÓRICO / LEGACY**. La ANN vigente no deriva de estos scripts ni sustituye sus registros: **ANN-V1-MODERN-ET0 = PASS**, con `FEATURE_SET_FULL / MLP [32]` y target `PM_FAO56_V1` calculada. El cierre moderno usa protocolo temporal, seeds congeladas y TEST `retrospective_locked`; sus métricas, límites y preservación están en [2.1 — ANN moderna](docs_2.1_Estimación%20Et0%20ANN%20%28tensorflow%29.md). No se interpreta como validación frente a ET₀ instrumental independiente.
