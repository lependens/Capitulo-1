commit 7058e640d0ec200a965cac8503b918eca9f5ec7e
Author: Codex <codex@openai.com>
Date:   Mon Nov 9 13:25:51 2026 +0100

    docs: close out ANN v1 modern ET0 results

diff --git "a/docs_1.2_Tratamiento de datos y c\303\241lculo de estimaciones.md" "b/docs_1.2_Tratamiento de datos y c\303\241lculo de estimaciones.md"
index 2a1dcfb..d9ff3f2 100755
--- "a/docs_1.2_Tratamiento de datos y c\303\241lculo de estimaciones.md"	
+++ "b/docs_1.2_Tratamiento de datos y c\303\241lculo de estimaciones.md"	
@@ -502,3 +502,13 @@ DEVELOPMENT 2004–2020 seleccionó y congeló 35 configuraciones entre 185 cand
 La recuperación final es **retrospective_recovery_evaluation**: intento original TEST_ACCESS_UNKNOWN y acceso técnico previo documentado. Modelos congelados antes de recovery; sin reselección, entrenamiento ni refit después de TEST. No afirmar primera/única apertura ni holdout prístino/prospectivo. FULL incluye variables cercanas a PM; aproximarla no demuestra exactitud frente a ET₀ instrumental.
 
 Dataset v1, baselines y AHC **PASS**; ML clásico **PASS**; ANN moderna **NOT STARTED**; WEB-V0.3-ML **READY**, sin despliegue en esta misión. Raw, Quality, resultados, contratos ejecutables y ciencia permanecen sin cambios. Run, hashes, receipt, snapshot REMOTE_VERIFIED y restore 77 archivos constan en [DATA_LOCATIONS](docs/DATA_LOCATIONS.md).
+
+---
+
+## ACTUALIZACIÓN 02/11/2026 — ANN moderna v1 publicada
+
+**ANN-V1-MODERN-ET0 = PASS.** Esta etapa sucede a los baselines, AHC y ML clásico; no reinterpreta sus resultados ni cambia raw, Quality o Dataset v1. La ANN seleccionada en DEVELOPMENT es **FEATURE_SET_FULL / MLP [32]**: una capa oculta de 32 neuronas, ReLU y Adam, con semillas 20261101–20261103. Su target es **PM_FAO56_V1**, referencia calculada: el resultado no es validación frente a ET₀ instrumental independiente.
+
+TEST 2021–2025, **retrospective_locked**, obtuvo n=15.958, MAE 0.037632, RMSE 0.051312, bias −0.022449 mm/día y R² 0.999100; cobertura 99.8748 %, 0 predicciones negativas y 0 no finitas. No se usó TEST para arquitectura, selección, entrenamiento ni refit; se abrió una vez tras congelar champion. No es un holdout globalmente prístino porque el periodo fue usado anteriormente en otros experimentos ML.
+
+Frente a HGB FULL, la ANN reduce RMSE ≈19.95 % y MAE ≈20.01 %, pero tiene bias negativo mayor. La documentación metodológica y el linaje de artefactos se conservan en [2.1 — ANN](docs_2.1_Estimación%20Et0%20ANN%20%28tensorflow%29.md); futuros trabajos deberán partir de validación prospectiva u otro contrato científico, no de reabrir esta selección.
