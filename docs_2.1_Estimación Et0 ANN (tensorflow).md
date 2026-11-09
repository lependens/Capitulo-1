commit 7058e640d0ec200a965cac8503b918eca9f5ec7e
Author: Codex <codex@openai.com>
Date:   Mon Nov 9 13:25:51 2026 +0100

    docs: close out ANN v1 modern ET0 results

diff --git "a/docs_2.1_Estimaci\303\263n Et0 ANN (tensorflow).md" "b/docs_2.1_Estimaci\303\263n Et0 ANN (tensorflow).md"
index 70d7ddb..f78966f 100755
--- "a/docs_2.1_Estimaci\303\263n Et0 ANN (tensorflow).md"	
+++ "b/docs_2.1_Estimaci\303\263n Et0 ANN (tensorflow).md"	
@@ -209,3 +209,37 @@ AHC-METHODOLOGY-AND-TEMPORAL-VALIDATION y PROJECT-ARTIFACT-BACKUP-V1 PASS; el co
 Siguiente orden: regresión lineal/regularizada, árboles, RF/extra trees, boosting, ANN después. TEST AHC retrospective_locked, con ajuste solo TRAIN y climatología separada, no test prospectivo nunca visto. No reutilizar AHC_CLIMATOLOGY como si acreditara métricas TEST.
 
 Snapshot remoto aceptado cubre un scope concreto; futuros resultados exclusivamente Windows requieren transferencia/ingestión explícita al servidor antes de preservación automática. [DATA_LOCATIONS](docs/DATA_LOCATIONS.md).
+
+---
+
+## ACTUALIZACIÓN 02/11/2026 — ANN moderna v1: cierre científico
+
+Las secciones anteriores describen ANN TensorFlow **histórica / legacy**. No son la implementación científica vigente ni se les atribuyen los resultados siguientes.
+
+**ANN-V1-MODERN-ET0 = PASS.** La campaña moderna aproxima **PM_FAO56_V1**, una referencia calculada; no demuestra precisión frente a ET₀ instrumental observada de manera independiente. El champion seleccionado solo con DEVELOPMENT 2004–2020 es **FEATURE_SET_FULL / MLP [32]**: una capa oculta, 32 neuronas, activación ReLU y optimizador Adam. Las semillas congeladas fueron `20261101`, `20261102` y `20261103`; el RMSE ponderado de DEVELOPMENT fue **0.074962 mm/día**.
+
+### Evaluación TEST y límites
+
+TEST 2021–2025 se abrió una vez después de congelar arquitectura/champion. No participó en arquitectura, selección, entrenamiento ni refit. Conserva etiqueta **retrospective_locked** y no es globalmente prístino porque ese periodo había sido usado antes en otros experimentos ML.
+
+| Métrica TEST | Valor |
+|---|---:|
+| n | 15.958 |
+| MAE | 0.037632 mm/día |
+| RMSE | 0.051312 mm/día |
+| bias | −0.022449 mm/día |
+| R² | 0.999100 |
+| cobertura | 99.8748 % |
+| predicciones negativas / no finitas | 0 / 0 |
+
+Ante HGB FULL (RMSE 0.064104; MAE 0.047045), la ANN reduce RMSE en **0.012791 mm/día** (≈19.95 %) y MAE en **0.009413 mm/día** (≈20.01 %). No hay una mejora sin matiz: el bias ANN, **−0.022449 mm/día**, es mayor en magnitud que el de HGB, **−0.000966 mm/día**. Mejor/peor año: 2024/2022 (RMSE 0.045078/0.058756); mejor/peor estación: IB04/IB02 (0.040756/0.065308).
+
+### Linaje y preservación
+
+- Pipeline/main: `754adceee12cc423ee06eb17b49e36429d25b70d`.
+- Artefacto: `/srv/siar-project/scientific-artifacts/ann-v1-modern-20261102T065642Z/`.
+- Manifest SHA256: `1978c8d018b57686b7e2d7fd9457ad2f9399a115e65789a13abd0f0c9359862f`.
+- Receipt: `science-ann-v1-modern-20261102T065642Z_447f9498.json`, estado `published-verified`.
+- Snapshot: `backup_20261102T171546Z_40b4cd67`, **REMOTE_VERIFIED**.
+
+Dataset v1, raw y Quality no fueron modificados. La siguiente evolución no presupone reabrir esta selección: requiere validación prospectiva y/o un nuevo contrato científico.
