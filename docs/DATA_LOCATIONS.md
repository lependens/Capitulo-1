commit 7058e640d0ec200a965cac8503b918eca9f5ec7e
Author: Codex <codex@openai.com>
Date:   Mon Nov 9 13:25:51 2026 +0100

    docs: close out ANN v1 modern ET0 results

diff --git a/docs/DATA_LOCATIONS.md b/docs/DATA_LOCATIONS.md
index a5de064..c51a548 100644
--- a/docs/DATA_LOCATIONS.md
+++ b/docs/DATA_LOCATIONS.md
@@ -247,3 +247,18 @@ Los intentos FAILED anteriores no se convierten en resultados válidos. Intento
 | Preservación | REMOTE_VERIFIED |
 
 Montaje de datos read-only; sin recálculo científico, cambios de artefactos científicos, Dataset ni raw. V0.2 queda como histórico/rollback. Paths completos, hash del paquete (distinto del Parquet) y evidencia en el acta, sin inspección nueva del servidor.
+
+---
+
+## ANN moderna v1 — resultado publicado 02/11/2026
+
+| Elemento | Identidad / ubicación |
+|---|---|
+| Estado | `ANN-V1-MODERN-ET0 = PASS` |
+| Artefacto canónico | `/srv/siar-project/scientific-artifacts/ann-v1-modern-20261102T065642Z/` |
+| Pipeline/main | `754adceee12cc423ee06eb17b49e36429d25b70d` |
+| Manifest SHA256 | `1978c8d018b57686b7e2d7fd9457ad2f9399a115e65789a13abd0f0c9359862f` |
+| Receipt | `science-ann-v1-modern-20261102T065642Z_447f9498.json` (`published-verified`) |
+| Snapshot | `backup_20261102T171546Z_40b4cd67` — `REMOTE_VERIFIED` |
+
+Este artefacto publica la ANN moderna v1 sobre `PM_FAO56_V1`; no modifica Dataset v1, raw ni Quality. La preservación remota no convierte la referencia calculada PM en una observación instrumental independiente.
