commit 7058e640d0ec200a965cac8503b918eca9f5ec7e
Author: Codex <codex@openai.com>
Date:   Mon Nov 9 13:25:51 2026 +0100

    docs: close out ANN v1 modern ET0 results

diff --git a/docs/ROADMAP.md b/docs/ROADMAP.md
index 4181570..2482a5f 100644
--- a/docs/ROADMAP.md
+++ b/docs/ROADMAP.md
@@ -9,13 +9,13 @@
 | Línea | Ya demostrado | Siguiente gate |
 |---|---|---|
 | Observaciones / Quality | POST corregido COMPLETE; 2.345 ausencias confirmadas, 0 PENDING_RETRY | Mantener snapshot cerrado; CATALOG-TZ-001 no bloqueante |
-| Ciencia / 05 | Dataset v1, cuatro baselines, AHC y ML clásico PASS | Conservar evaluación recuperada/limitaciones; ANN moderna NOT STARTED, siguiente fase bajo contrato propio |
+| Ciencia / 05 | Dataset v1, cuatro baselines, AHC, ML clásico y ANN moderna v1 PASS | Conservar límites retrospectivos; siguiente fase: validación prospectiva o nuevo contrato científico |
 | Preservación / 02B | PROJECT-ARTIFACT-BACKUP-V1 PASS / MAINTAINED; Windows ingestion automation PASS | Productor finaliza/prepara/firma READY; flujo posterior automatizado; mantener verificación y recuperabilidad |
 | SIAR Sync / 02 | v2.3.3 canónica, snapshot científico 2025 cerrado | HOLD → production alignment + incremental 2026 demostrado |
 | Forecast SIAR / 02 | Timer ACTIVE; reconciliador enabled/active cada 15 min; OAuth Production | Nuevo run automático → backup automático → COMPLETE remoto → CAPTURE RUNNING; mínimo 2 ciclos PILOT RUNNING |
 | AEMET Track A / 02 | E2E DEMONSTRATED; capture disabled/inactive; nuevo backup automático NOT ACTIVE | Reactivar tras SIAR-FORECAST-CAPTURE RUNNING, sin dependencia de Sync |
 | AEMET Track B | Investigación independiente OPEN | Identificar feed exacto usado por SIAR |
-| Producto / 06 | WEB-V0.3 Scientific Explorer PASS / DEPLOYED en LAN | ANN moderna y/o validación prospectiva; consumo web tras publicación, sin V0.4 inmediato |
+| Producto / 06 | WEB-V0.3 Scientific Explorer PASS / DEPLOYED en LAN | Consumir ANN v1 publicada y/o validación prospectiva; sin V0.4 inmediato |
 
 **La captura prospectiva no espera a los modelos ni a reactivar SIAR Sync.** No reconstruir emisiones perdidas durante la pausa. El incidente Toshiba está CLOSED_OPERATIONALLY / MONITORING; no es una nueva fase activa.
 
@@ -86,7 +86,7 @@ Conserva 75.496 observaciones raw, 23 columnas, una fila exterior, ledger y prov
 
 HG-Rs tiene mejor RMSE/R² que VAL y VAL mejor MAE/bias en la cohorte común; no hay vencedor universal. La mayor similitud de inputs con EtPMon puede favorecer concordancia, sin causalidad demostrada. Separar accuracy against EtPMon de input requirements / operational availability; sigue importando precisión con menos variables. EtPMon no es ET₀ instrumental independiente.
 
-## Fase 6 — AHC y ML clásico cerrados; ANN moderna no iniciada
+## Fase 6 — AHC, ML clásico y ANN moderna cerrados
 
 ✅ **AHC-METHODOLOGY-AND-TEMPORAL-VALIDATION PASS**, PR #38/#39. OLS0 por estación a=Σ(x·PM)/Σ(x²), referencia primaria PM_FAO56_V1; EtPMon secundaria calculada. TRAIN 2004–2015, VALIDATION 2016–2020, TEST 2021–2025; ajuste solo TRAIN, sin refit TRAIN+VALIDATION. TEST retrospective_locked, no prospectivo nunca visto.
 
@@ -105,7 +105,8 @@ Estado científico y de implementación:
 - ✅ Linux smoke, solo infraestructura; sin selección científica ni TEST.
 - ✅ Runtime rolling oficial (PR #48), comando `ml-rolling-campaign`.
 - ✅ **ML-V1-CLASSICAL-MODELS PASS**, cierre de evaluación recuperada; [explicación y resultados](../docs_1.2.6_Modelos%20ML%20cl%C3%A1sicos%20para%20estimaci%C3%B3n%20de%20ET0.md).
-- ⚪ ANN moderna NOT STARTED; su nueva fase requiere contrato/objetivo propio.
+- ✅ **ANN-V1-MODERN-ET0 PASS**: MLP [32] FULL seleccionado en DEVELOPMENT; TEST retrospective_locked publicado sin reselección, entrenamiento ni refit.
+- ⚪ Próximas fases: validación prospectiva y nuevas preguntas científicas bajo contrato propio; ANN ya no es un gate pendiente.
 
 [Protocolo rolling vigente](../scientific_pipeline/docs/ml_validation_contract_refinement.md): selección en DEVELOPMENT F1–F4, congelación y refit DEVELOPMENT antes de TEST. La recuperación efectiva conserva TEST_ACCESS_UNKNOWN del intento original y la etiqueta retrospective_recovery_evaluation; no afirmar apertura única ni holdout prístino. Conservar `retrospective_locked`; no aplicar este refit al AHC ya publicado. El intento FAILED y el smoke no sirven para seleccionar ciencia. No reabrir baselines ni metodología AHC.
 
@@ -154,4 +155,10 @@ PM/HS/VAL y Dataset v1 intactos. AHC y backup ya están PASS; siguientes experim
 
 ## Cierre vigente de producto — 01/11/2026
 
-**WEB-V0.3-SCIENTIFIC-EXPLORER = PASS / DEPLOYED** (PR #57–#60). [Acta canónica](../web_app/docs/release-v03.md): [LAN](http://192.168.1.43:8088/), 43 modelos (35 ML), 12 estaciones; comparación de modelos, análisis por estación y temporal, residuales, feature sets, fichas de modelos, glosario científico y audit/provenance. Montajes de datos read-only. WEB recalculates science = NO; scientific artifacts modified = NO; Dataset modified = NO; raw modified = NO. V0.2 queda como histórico/rollback. No se requiere WEB-V0.4 inmediato: siguiente avance científico ANN moderna y/o validación prospectiva; la web los consumirá después de publicación.
+**WEB-V0.3-SCIENTIFIC-EXPLORER = PASS / DEPLOYED** (PR #57–#60). [Acta canónica](../web_app/docs/release-v03.md): [LAN](http://192.168.1.43:8088/), 43 modelos (35 ML), 12 estaciones; comparación de modelos, análisis por estación y temporal, residuales, feature sets, fichas de modelos, glosario científico y audit/provenance. Montajes de datos read-only. WEB recalculates science = NO; scientific artifacts modified = NO; Dataset modified = NO; raw modified = NO. V0.2 queda como histórico/rollback. No se requiere WEB-V0.4 inmediato: ANN v1 ya es ciencia publicada; la web podrá consumirla junto con validación prospectiva tras publicación.
+
+---
+
+## Cierre ANN v1 — 02/11/2026
+
+✅ **ANN-V1-MODERN-ET0 = PASS.** Champion `FEATURE_SET_FULL / MLP [32]`; target `PM_FAO56_V1`; TEST retrospective_locked n=15.958, RMSE 0.051312 mm/día. La ANN no es validación instrumental independiente y su bias −0.022449 mm/día debe leerse junto a la mejora de RMSE/MAE frente a HGB FULL. La siguiente decisión científica no es reabrir selección: es validación prospectiva y/o una nueva pregunta con contrato propio. [Documentación ANN](../docs_2.1_Estimación%20Et0%20ANN%20%28tensorflow%29.md).
