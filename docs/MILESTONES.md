commit 7058e640d0ec200a965cac8503b918eca9f5ec7e
Author: Codex <codex@openai.com>
Date:   Mon Nov 9 13:25:51 2026 +0100

    docs: close out ANN v1 modern ET0 results

diff --git a/docs/MILESTONES.md b/docs/MILESTONES.md
index 2031633..46f25ef 100644
--- a/docs/MILESTONES.md
+++ b/docs/MILESTONES.md
@@ -92,6 +92,10 @@ Se añaden cierres posteriores a las entradas que declaraban estas misiones pend
 
 | 2026-11-01 | WINDOWS-SCIENCE-INGEST-AUTOMATION | PASS; E2E con artefacto sintético 47 bytes → snapshot backup_20261101T065239Z_71bbe72e REMOTE_VERIFIED; idempotencia new=0/unchanged=1 PASS; failure/retry PASS. | Cierre operativo 02B comunicado por Dirección 00; [contrato Windows](backup/windows-ingestion.md) | Supersede operacionalmente DESIGN_READY / STILL_PENDING del 31/10, sin borrarlo. Finalización y READY firmado explícitos; transferencia/verificación/ingestión/preservación automáticas. PROJECT-ARTIFACT-BACKUP-V1 PASS / MAINTAINED; no es resultado científico. |
 
-| 2026-11-01 | ML-V1-CLASSICAL-MODELS | PASS autorizado por Dirección 00; champion DEVELOPMENT HGB FULL, TEST común n=15.958, RMSE 0.064104 mm/día contra PM calculado. | [Explicación 1.2.6](../docs_1.2.6_Modelos%20ML%20cl%C3%A1sicos%20para%20estimaci%C3%B3n%20de%20ET0.md); run ml-recovery-c2c5c34e7cc642a181d0f3ee12a5ef7e, pipeline 94ebf67df2954e6818fc5f70c6b2e91b1b07f505 | Evaluación retrospective_recovery_evaluation, sin reselección/entrenamiento/refit después de TEST; no holdout prístino ni validación instrumental. ANN moderna NOT STARTED; WEB-V0.3-ML READY. Snapshot backup_20261101T095440Z_0381c133 REMOTE_VERIFIED y restore 77 archivos PASS. |
+| 2026-11-01 | ML-V1-CLASSICAL-MODELS | PASS autorizado por Dirección 00; champion DEVELOPMENT HGB FULL, TEST común n=15.958, RMSE 0.064104 mm/día contra PM calculado. | [Explicación 1.2.6](../docs_1.2.6_Modelos%20ML%20cl%C3%A1sicos%20para%20estimaci%C3%B3n%20de%20ET0.md); run ml-recovery-c2c5c34e7cc642a181d0f3ee12a5ef7e, pipeline 94ebf67df2954e6818fc5f70c6b2e91b1b07f505 | Evaluación retrospective_recovery_evaluation, sin reselección/entrenamiento/refit después de TEST; no holdout prístino ni validación instrumental. ANN moderna NOT STARTED **(estado histórico al 01/11, superado el 02/11)**; WEB-V0.3-ML READY. Snapshot backup_20261101T095440Z_0381c133 REMOTE_VERIFIED y restore 77 archivos PASS. |
 
 | 2026-11-01 | WEB-V0.3-SCIENTIFIC-EXPLORER | PASS / DEPLOYED LAN; 43 modelos, 35 ML, 12 estaciones; sin recálculo científico ni cambio Dataset/raw. | PR #57/#58/#59/#60; [acta canónica](../web_app/docs/release-v03.md) | V0.2 histórico/rollback; artefacto web-derived-v03-explorer-20261101-v1, snapshot backup_20261101T123421Z_8022d3fd REMOTE_VERIFIED. Próxima ciencia: ANN moderna y/o validación prospectiva, consumo web posterior. |
+
+---
+
+| 2026-11-02 | ANN-V1-MODERN-ET0 | PASS: `FEATURE_SET_FULL / MLP [32]`, una capa ReLU de 32 neuronas y Adam; TEST n=15.958, MAE 0.037632, RMSE 0.051312, bias −0.022449, R² 0.999100; 0 negativas/no finitas. | Pipeline/main `754adceee12cc423ee06eb17b49e36429d25b70d`; manifest `1978c8d018b57686b7e2d7fd9457ad2f9399a115e65789a13abd0f0c9359862f`; receipt `science-ann-v1-modern-20261102T065642Z_447f9498.json`; snapshot `backup_20261102T171546Z_40b4cd67` REMOTE_VERIFIED | ANN aproxima PM_FAO56_V1 calculada, no ET₀ instrumental. Champion congelado antes de TEST; TEST retrospective_locked y no globalmente prístino. Mejora RMSE/MAE sobre HGB FULL pero conserva bias negativo mayor. |
