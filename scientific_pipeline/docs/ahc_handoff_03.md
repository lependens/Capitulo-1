# Handoff científico 05 → 03 — 24/10/2026

> **Estado posterior de preservación — 25/10/2026.** El handoff del 24/10 conserva la situación de backup al publicar la ciencia. Después PROJECT-ARTIFACT-BACKUP-V1 cerró PASS: snapshot backup_20261024T215639Z_63d0ef01 REMOTE_VERIFIED, 128 archivos en el scope aceptado. La afirmación inferior de backup pendiente es histórica; [alcance y límites vigentes](../../docs/DATA_LOCATIONS.md). No se modifican sus métricas ni manifests.

AHC-METHODOLOGY-AND-TEMPORAL-VALIDATION = PASS.
Autoridad: [resultados, tablas y hashes](../results/ahc-20261024T172400Z/README.md).
Código PR #38; pipeline `38503dbe8422982deeb3680bbc3e82fd950da4e1`.
Raw/Quality/Dataset v1 permanecen congelados. No ANN, ML entrenado ni Web.

La definición v1 usa OLS por origen por estación contra PM_FAO56_V1;
mean-ratio legacy queda histórico y sensibilidad TRAIN. TRAIN 2004–2015,
VALIDATION 2016–2020, TEST 2021–2025; ningún refit. TEST retrospectivo,
no confirmación prospectiva. EtPMon permanece referencia calculada secundaria.

TEST común n=15.958: RMSE anual ajustado HS 0.638838, HG-Rs 0.414241,
VAL 0.523955 mm/día; HG-Rs estacional, seleccionado solo en VALIDATION,
0.401392. Estabilidad de bloques PASS, CV máximo 6,61 %; ninguna causa
física inferida de estos cambios. Hay empeoramientos estación-año que se
conservan y documentan, especialmente HS en IB11/2025.

AHC_CLIMATOLOGY se calcula después de TEST usando el histórico hasta 2025,
con identidad separada. IB101 solo descriptivo, deployable=false; IB09 FAIL
sin ajuste; IB07 sin validación/test después de su cierre. No presentar
climatología como rendimiento fuera de muestra ni modificar resultados legacy.

Contratos [ML](ml_experiment_contract.md) y [visualización](visualization_contract.md)
definidos. Futuros experimentos ML requieren presupuesto/semillas/selección
congelados antes de entrenar; este TEST ya publicado debe seguir etiquetado
retrospectivo. Una interfaz futura consume tablas y métricas, sin recalcular ciencia.

Manifest TEST SHA256:
`502d7a39b82c8b5032931370ae8f05f4d8e7ecbbb93f6946810d0e47434fc469`.
Manifest cierre SHA256:
`39be9d706f962722862c50ce6b25fad5d0e93a57d954c4b71aa77cfe5942d6cd`.
Los outputs masivos están fuera de Git en el workspace Windows, indexados por
hash; PROJECT-ARTIFACT-BACKUP-V1 sigue pendiente de backup independiente.
No afirmar réplica íntegra en servidor Linux ni backup completo por este handoff.
