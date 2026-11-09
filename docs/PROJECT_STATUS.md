commit 7058e640d0ec200a965cac8503b918eca9f5ec7e
Author: Codex <codex@openai.com>
Date:   Mon Nov 9 13:25:51 2026 +0100

    docs: close out ANN v1 modern ET0 results

diff --git a/docs/PROJECT_STATUS.md b/docs/PROJECT_STATUS.md
index e06f3f6..d7aa199 100644
--- a/docs/PROJECT_STATUS.md
+++ b/docs/PROJECT_STATUS.md
@@ -1,6 +1,6 @@
 # PROJECT STATUS — SIAR / ET₀
 
-**Actualizado: 2026-11-01 (Europe/Madrid), cierre científico ML clásico y preservación vigente.** Base releída: `90a50c1d6e8ef6563798fdec10cb253f80f0a698`. La evidencia de ingestión, preservación remota y restore procede de la comprobación operativa del servidor; esta actualización documental no modifica los artefactos científicos. Un runtime fusionado no demuestra una campaña científica completada; un PASS histórico no acredita que un timer siga ejecutándose hoy.
+**Actualizado: 2026-11-02 (Europe/Madrid), cierre ANN v1 moderna y preservación vigente.** Base releída: `754adceee12cc423ee06eb17b49e36429d25b70d`. La evidencia de ingestión, preservación remota y restore procede de la comprobación operativa del servidor; esta actualización documental no modifica los artefactos científicos. Un runtime fusionado no demuestra una campaña científica completada; un PASS histórico no acredita que un timer siga ejecutándose hoy.
 
 ## Cómo leer el estado
 
@@ -18,14 +18,14 @@
 |---|---|---|
 | Quality SIAR Baleares | COMPLETE; POST corregido publicado mediante PR #29; 12 estaciones | CATALOG-TZ-001 OPEN / NON_BLOCKING; completitud observacional distinta de completitud de auditoría |
 | Dataset Observational v1 | Construido/validado PASS; SHA256 verificado en snapshot remoto aceptado | Conservar identidad, provenance y acceso a la copia completa; no se afirma que el JSON masivo esté en Git |
-| Ciencia ET₀ | Dataset v1, cuatro baselines, AHC y ML-V1-CLASSICAL-MODELS PASS | Evaluación retrospectiva recuperada; referencia PM calculada; ANN moderna NOT STARTED |
+| Ciencia ET₀ | Dataset v1, cuatro baselines, AHC, ML clásico y ANN-V1-MODERN-ET0 PASS | Los modelos aproximan PM_FAO56_V1 calculada; TEST retrospective_locked, no validación instrumental independiente |
 | Preservación científica | PROJECT-ARTIFACT-BACKUP-V1 PASS / MAINTAINED; Windows ingestion PASS | Finalización científica y preparación/READY firmado explícitos; transferencia, verificación servidor, ingestión y preservación posteriores automáticas |
 | SIAR Sync | v2.3.3 canónica; SIAR-SYNC-RUNTIME = HOLD | SIAR-SYNC-PRODUCTION-ALIGNMENT + SIAR-SYNC-INCREMENTAL-2026 |
 | SIAR Forecast | Timer de captura ACTIVE; reconciliador enabled/active cada 15 min, último Result success | Nuevo run real automático → backup automático → COMPLETE remoto; después CAPTURE RUNNING, mínimo dos ciclos PILOT RUNNING |
 | AEMET Track A | E2E DEMONSTRATED; capture timer disabled/inactive; nuevo backup automático NOT ACTIVE | Reactivar tras SIAR-FORECAST-CAPTURE RUNNING; no depende de SIAR Sync |
 | AEMET Track B | OPEN | Investigar feed AEMET exacto utilizado por SIAR, independientemente de Track A |
 | Storage | Incidente CLOSED_OPERATIONALLY / MONITORING; preservación PASS; SSD IN SERVICE | Reabrir solo con nueva evidencia objetiva; sin plan de sustitución del SSD |
-| Producto / 06 | WEB-V0.3 Scientific Explorer PASS / DEPLOYED en LAN | ANN moderna y/o validación prospectiva; consumo web tras publicación, sin V0.4 inmediato |
+| Producto / 06 | WEB-V0.3 Scientific Explorer PASS / DEPLOYED en LAN | Consumir resultados ANN publicados y/o validación prospectiva; sin V0.4 inmediato |
 
 ## Observaciones y Quality: cierre vigente
 
@@ -142,7 +142,7 @@ PR [#46](https://github.com/lependens/Capitulo-1/pull/46) y [#47](https://github
 
 **ML-V1-TEST-RECOVERY = EXECUTED / PRESERVED.** Pipeline `94ebf67df2954e6818fc5f70c6b2e91b1b07f505`; los 35 modelos y la selección permanecen congelados. TEST se abrió en la recuperación y existen predicciones y métricas; no hubo nuevo entrenamiento, selección ni refit. Dataset SHA256 `0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb`; protocolo `45f1c4f3ed20bb70dc3b1ca06a8e23ec14f95aae6b60f574e44a8a261cc07048`; manifest AHC TEST `502d7a39b82c8b5032931370ae8f05f4d8e7ecbbb93f6946810d0e47434fc469`; selection `a41ee87956368d6c0e1fae2b39177ffe8c42cae7658477172c46725409f829c7`; bundle `b1be36a43ff465cbf8662e6182c3c862037d1bc6bbc8295320013c99e7893946`; manifest de run `05834edaebf3f990729ebb8b15ecd7511bb96e006bfab69f11408a250f432aab`. Artefacto canónico: `/srv/siar-project/scientific-artifacts/ml-v1-test-recovery-20261101T090828Z/`. La ingesta incremental informó `new=1, unchanged=2`, receipt `science-ml-v1-test-recovery-20261101T090828Z_83cf4565.json` con estado `published-verified`. Snapshot `backup_20261101T095440Z_0381c133`, manifest SHA256 `f3cfa2c4d1d7d4197d5495f187275dae9d2784eabb1d6b952b0bb44db796fd46`, **REMOTE_VERIFIED** y restore en cuarentena **RESTORE_HASHES_PASS (77 archivos)**. **ML-V1-CLASSICAL-MODELS = PASS**, cierre científico autorizado por Dirección 00 tras revisión. Champion elegido en DEVELOPMENT: **ML_HIST_GRADIENT_BOOSTING__FEATURE_SET_FULL**; TEST común **n=15.958**, MAE **0.047045**, RMSE **0.064104**, bias **-0.000966 mm/día**, R² **0.998595**, cobertura **99.875 %**. La explicación de métodos, inputs, comparación AHC y estabilidad está en [1.2.6](../docs_1.2.6_Modelos%20ML%20cl%C3%A1sicos%20para%20estimaci%C3%B3n%20de%20ET0.md).
 
-Intento original TEST_ACCESS_UNKNOWN; final **retrospective_recovery_evaluation**. Modelos congelados antes del acceso de recuperación, sin reselección/entrenamiento/refit posteriores. No es una única apertura TEST ni validación instrumental independiente de ET₀. Se conservan 1.007 predicciones negativas de regresiones entre 35 modelos, sin clipping; champion 0 negativas/0 no finitas. **ANN moderna NOT STARTED; WEB-V0.3 Scientific Explorer PASS / DEPLOYED.**
+Intento original TEST_ACCESS_UNKNOWN; final **retrospective_recovery_evaluation**. Modelos congelados antes del acceso de recuperación, sin reselección/entrenamiento/refit posteriores. No es una única apertura TEST ni validación instrumental independiente de ET₀. Se conservan 1.007 predicciones negativas de regresiones entre 35 modelos, sin clipping; champion 0 negativas/0 no finitas. **ANN-V1-MODERN-ET0 = PASS; WEB-V0.3 Scientific Explorer PASS / DEPLOYED.**
 
 ## SIAR Sync: HOLD independiente del snapshot científico
 
@@ -219,8 +219,8 @@ Toshiba ya no es la única copia **para los artefactos científicos y SIAR publi
 
 ## Próximos pasos
 
-1. **05 / ciencia:** ML-V1-CLASSICAL-MODELS PASS; conservar resultados, protocolo y límites del cierre. ANN moderna NOT STARTED; cualquier fase nueva requiere su contrato y autorización, sin reutilizar TEST para reseleccionar.
-2. **06 / producto:** mantener WEB-V0.3 Scientific Explorer desplegada; consumir ANN moderna y/o validación prospectiva cuando estén publicadas, sin recalcular ciencia ni exigir V0.4 inmediato.
+1. **05 / ciencia:** ML-V1-CLASSICAL-MODELS y ANN-V1-MODERN-ET0 PASS; conservar resultados, protocolo y límites de ambos cierres. Próximas fases requieren contrato y autorización, sin reutilizar TEST para reseleccionar.
+2. **06 / producto:** mantener WEB-V0.3 Scientific Explorer desplegada; consumir la ANN publicada y/o validación prospectiva cuando corresponda, sin recalcular ciencia ni exigir V0.4 inmediato.
 3. **Productor / 02B:** finalizar explícitamente cada resultado, preparar y firmar READY y mantener precondiciones de ssh-agent. Windows automation PASS; transferencia/verificación/ingestión/preservación automáticas y reconciliación diaria enabled/active. WEB-V0.2 conserva su receipt/snapshot REMOTE_VERIFIED.
 4. Infraestructura: gate SIAR Forecast de nuevo run real con backup/COMPLETE, AEMET tras CAPTURE RUNNING y SIAR Sync alignment/incremental 2026 permanecen separados, sin cambios en esta misión.
 5. Issue/valid time/D+n, ET₀+Kc/ETc/riego y validación prospectiva mantienen sus gates propios.
@@ -247,4 +247,16 @@ Dataset y todos los resultados PM/HS/VAL anteriores intactos, sin calibración.
 
 ## Cierre vigente de producto — 01/11/2026
 
-**WEB-V0.3-SCIENTIFIC-EXPLORER = PASS / DEPLOYED** (PR #57–#60). [Acta canónica](../web_app/docs/release-v03.md): [LAN](http://192.168.1.43:8088/), 43 modelos (35 ML), 12 estaciones; comparación de modelos, análisis por estación y temporal, residuales, feature sets, fichas de modelos, glosario científico y audit/provenance. Montajes de datos read-only. WEB recalculates science = NO; scientific artifacts modified = NO; Dataset modified = NO; raw modified = NO. V0.2 queda como histórico/rollback. No se requiere WEB-V0.4 inmediato: siguiente avance científico ANN moderna y/o validación prospectiva; la web los consumirá después de publicación.
+**WEB-V0.3-SCIENTIFIC-EXPLORER = PASS / DEPLOYED** (PR #57–#60). [Acta canónica](../web_app/docs/release-v03.md): [LAN](http://192.168.1.43:8088/), 43 modelos (35 ML), 12 estaciones; comparación de modelos, análisis por estación y temporal, residuales, feature sets, fichas de modelos, glosario científico y audit/provenance. Montajes de datos read-only. WEB recalculates science = NO; scientific artifacts modified = NO; Dataset modified = NO; raw modified = NO. V0.2 queda como histórico/rollback. No se requiere WEB-V0.4 inmediato: ANN v1 ya está publicada; la web podrá consumirla junto con futura validación prospectiva después de sus publicaciones.
+
+---
+
+## ANN moderna v1 — cierre científico 02/11/2026
+
+**ANN-V1-MODERN-ET0 = PASS.** La ANN aproxima el target calculado **PM_FAO56_V1**; no acredita precisión frente a ET₀ instrumental independiente. Champion congelado en DEVELOPMENT: **FEATURE_SET_FULL / MLP [32]**, una capa oculta de 32 neuronas, ReLU y Adam, con semillas `20261101`, `20261102` y `20261103`.
+
+DEVELOPMENT obtuvo RMSE ponderado **0.074962 mm/día**. En TEST **retrospective_locked** 2021–2025: n=**15.958**, MAE **0.037632**, RMSE **0.051312**, bias **−0.022449 mm/día**, R² **0.999100** y cobertura **99.8748 %**; 0 predicciones negativas y 0 no finitas. La ANN no usó TEST para arquitectura, selección, entrenamiento ni refit, y lo abrió una vez después de congelar el champion. El periodo no es globalmente prístino porque fue utilizado antes en experimentos ML.
+
+Frente al benchmark HGB FULL (RMSE 0.064104; MAE 0.047045), reduce RMSE **0.012791 mm/día** (≈19.95 %) y MAE **0.009413 mm/día** (≈20.01 %). Su bias negativo es mayor que HGB (−0.022449 frente a −0.000966 mm/día), limitación explícita. Mejor/peor año: 2024 / 2022 (RMSE 0.045078 / 0.058756); mejor/peor estación: IB04 / IB02 (0.040756 / 0.065308).
+
+Artefacto canónico: `/srv/siar-project/scientific-artifacts/ann-v1-modern-20261102T065642Z/`; manifest SHA256 `1978c8d018b57686b7e2d7fd9457ad2f9399a115e65789a13abd0f0c9359862f`; receipt `science-ann-v1-modern-20261102T065642Z_447f9498.json`; snapshot `backup_20261102T171546Z_40b4cd67`, **REMOTE_VERIFIED**. El pipeline/main asociado es `754adceee12cc423ee06eb17b49e36429d25b70d`. Véase [documentación ANN](../docs_2.1_Estimación%20Et0%20ANN%20%28tensorflow%29.md).
