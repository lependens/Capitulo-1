# 1.2.6 — Modelos ML clásicos para estimación de ET₀

**Actualización: 01/11/2026. ML-V1-CLASSICAL-MODELS = PASS**, cierre científico autorizado por Dirección 00. Este documento explica los resultados comunicados del run preservado; no ejecuta modelos ni recalcula métricas. Los contratos detallados siguen en [scientific_pipeline/docs/](scientific_pipeline/docs/ml_v1_classical_methodology.md); los bytes y evidencias externas, en [DATA_LOCATIONS](docs/DATA_LOCATIONS.md).

## De ecuaciones explícitas a relaciones aprendidas

El proyecto distingue tres familias metodológicas. Compartir unidades de salida (mm/día) no las convierte en métodos equivalentes.

| Familia | Métodos | Qué se fija y qué se aprende |
|---|---|---|
| Formulaciones deterministas/físicas o empíricas | PM_FAO56_V1, HS_RA_V1, HG_RS_V1, VALIANTZAS_V1 | Ecuaciones explícitas y decisiones numéricas versionadas; los cuatro baselines v1 no se calibran localmente |
| Calibración | AHC_VALIDATED | Aprende un multiplicador de un método simplificado frente a PM, con ajuste y validación temporal separados |
| Modelos ML aprendidos | Linear, Ridge, Elastic Net, Decision Tree, Random Forest, Extra Trees, HistGradientBoosting | Aprenden una relación entre predictores y target dentro de un espacio experimental predefinido |
| Aprendizaje posterior | ANN moderna | NOT STARTED; los prototipos legacy no son una ANN v1 validada |

PM_FAO56_V1 es la referencia física principal del proyecto. HS-Ra simplifica con temperatura/geometría solar; HG-Rs utiliza radiación medida; Valiantzas v1 incorpora su formulación empírica aprobada. AHC ajusta una salida ya calculada. ML aprende coeficientes, particiones o ensembles desde los ejemplos, con complejidad y variables controladas. Linear modela una relación lineal; Ridge y Elastic Net añaden regularización; Decision Tree divide el espacio de entradas; Random Forest y Extra Trees combinan árboles; HistGradientBoosting construye un ensemble por boosting. Sus diferencias se evalúan temporalmente, no solo por capacidad de ajuste.

**Target ML: PM_FAO56_V1.** Se reproduce/aproxima una referencia calculada PM, no una medición instrumental independiente de ET₀. El alto acuerdo con PM no demuestra por sí solo exactitud frente a ET₀ real. EtPMon también es una referencia calculada SIAR. Los resultados clásicos frente a EtPMon y los ML frente a PM deben leerse con su referencia explícita.

## Protocolo temporal: selección antes de TEST

Dataset Observational v1 permanece congelado. DEVELOPMENT comprende **2004–2020**, con cuatro folds causales expansivos:

| Fold | Entrenamiento | Validación posterior |
|---|---|---|
| F1 | 2004–2009 | 2010–2012 |
| F2 | 2004–2012 | 2013–2015 |
| F3 | 2004–2015 | 2016–2018 |
| F4 | 2004–2018 | 2019–2020 |

En cada fold, modelo y transformaciones se ajustan exclusivamente en su TRAIN. Las validaciones posteriores seleccionan candidatos mediante la media de RMSE ponderada por observaciones elegibles; se informa además dispersión y rango. Después se congelan las configuraciones y el champion seleccionado en DEVELOPMENT, y se hace refit con DEVELOPMENT completo, **antes de cualquier acceso de recuperación a TEST**.

TEST comprende **2021–2025**, con etiqueta **retrospective_locked**. No se utilizó para elegir algoritmos, variables o hiperparámetros, ni para entrenar/refit los modelos recuperados. El mejor valor observado en TEST no sustituye al champion elegido en desarrollo. La historia técnica de accesos impide afirmar una única apertura: véase el apartado de recuperación.

El protocolo ML rolling no sustituye el protocolo AHC publicado: AHC se ajustó solo en TRAIN 2004–2015 y no se refiteó con TRAIN+VALIDATION. [Refinamiento temporal](scientific_pipeline/docs/ml_validation_contract_refinement.md).

## Espacio experimental y disponibilidad de inputs

Cinco familias meteorológicas se cruzan con siete algoritmos: **185 candidatos**, evaluados en F1–F4, y **35 configuraciones congeladas** (una por celda algoritmo × familia). No son solo 185 ajustes: cada candidato atraviesa los cuatro folds.

| Familia abreviada | ID | Meteorología utilizada |
|---|---|---|
| T | FEATURE_SET_T | Tcalc y TD |
| T+RH | FEATURE_SET_T_RH | Tcalc, TD y humedad media/mínima/máxima |
| T+Rs | FEATURE_SET_T_RS | Tcalc, TD y radiación medida |
| T+RH+Rs | FEATURE_SET_T_RH_RS | Temperatura, humedad y radiación |
| FULL | FEATURE_SET_FULL | Temperatura, humedad, radiación y viento |

Tcalc=(Tmax+Tmin)/2 y TD=Tmax−Tmin. Todas las familias incluyen además contexto declarado: latitud, altitud, seno/coseno anual y Ra geométrica. Por tanto, T no significa «únicamente una columna de temperatura». **station_code es clave, no predictor** numérico ni categórico. PM target, otros baselines, AHC y residuos no entran como features. Las regresiones usan escalado aprendido solo en su ventana de ajuste; árboles no lo necesitan. Casos completos por familia, sin imputación meteorológica ni clipping.

Los siete algoritmos son Linear, Ridge, Elastic Net, Decision Tree, Random Forest, Extra Trees y HistGradientBoosting. La [configuración congelada](scientific_pipeline/experiments/ml-v1-classical.json) es autoridad para rejillas, semillas, elegibilidad y selección. Precisión y disponibilidad de inputs se reportan como dimensiones distintas.

## Champion elegido en DEVELOPMENT

**ML_HIST_GRADIENT_BOOSTING__FEATURE_SET_FULL**.

Candidate ID:
`mlc-95d2be33ea8867bcd84cb95281d7b7317e05fc02b4100e285a34df12e6e0647a`

| Parámetro | Valor |
|---|---:|
| learning_rate | 0.1 |
| max_iter | 200 |
| max_leaf_nodes | 31 |
| l2_regularization | 0.0 |

DEVELOPMENT rolling: **weighted mean RMSE=0.069255**, **weighted SD=0.003611**, rango de folds **0.063347–0.071917 mm/día**. Quedó **1.º de las 35 celdas seleccionadas** en desarrollo; esa identidad no se cambia después de mirar TEST.

## Evaluación TEST y comparación con AHC

Cohorte común **n=15.958**; SHA256 de claves:
`584d68d6dfc20799ea536562fc5f116a7f0ce166ac9f7ee7b95a9d44c255b59f`.

Error=predicción−PM_FAO56_V1. MAE, RMSE y bias en mm/día; R² predictivo=1−SSE/SST.

| Modelo sobre cohorte comparable | MAE | RMSE | bias | R² |
|---|---:|---:|---:|---:|
| HG_RS_AHC_VALIDATED_SELECTED | 0.306919 | 0.401392 | -0.011504 | 0.944921 |
| Champion ML HGB FULL | 0.047045 | 0.064104 | -0.000966 | 0.998595 |

Cobertura del champion: **15.958 / 15.978 = 99.875 %**. Se registran **0 predicciones negativas** y **0 no finitas** en el champion.

Frente al AHC seleccionado: **ΔRMSE=-0.337288 mm/día**, reducción relativa **≈84.0 %**; **ΔMAE=-0.259874 mm/día**; **ΔR²=+0.053674**. Son diferencias comunicadas del cierre, no cálculos nuevos en esta PR. La comparación mantiene la identidad del AHC estacional seleccionado y no la confunde con el benchmark anual ni con AHC_CLIMATOLOGY.

FULL utiliza temperatura, humedad, radiación y viento, variables estrechamente relacionadas con PM. Esa riqueza de inputs ayuda a interpretar la concordancia; no demuestra validación independiente de ET₀ real, transferencia geográfica o precisión prospectiva.

## Complejidad y ganancia con distintos inputs

Comparación TEST comunicada para FULL:

| Algoritmo FULL | RMSE, mm/día |
|---|---:|
| Linear | 0.266469 |
| Random Forest | 0.108895 |
| Extra Trees | 0.110261 |
| HistGradientBoosting | 0.064104 |

El champion reduce RMSE **≈75.9 % frente a Linear FULL**. La mejora describe esta tarea/dataset/protocolo, no una superioridad universal de boosting.

Para HistGradientBoosting, la comparación entre familias es:

| Familia | RMSE TEST, mm/día |
|---|---:|
| T | 0.546691 |
| T+RH | 0.429975 |
| T+Rs | 0.302158 |
| T+RH+Rs | 0.236410 |
| FULL | 0.064104 |

Destaca la fuerte ganancia **asociada a añadir viento** al conjunto completo. Esta tabla es comparación de requisitos de información/modelos, no una importancia causal ni una intervención experimental sobre el clima. No atribuir a una variable diferencias producidas por cohortes distintas; mantener cohortes propias para disponibilidad y comunes para comparación. Las importancias predictivas y sus dependencias con variables correlacionadas tampoco prueban causalidad.

## Estabilidad y comportamiento de predicción

| Año TEST | RMSE champion, mm/día |
|---|---:|
| 2021 | 0.06080 |
| 2022 | 0.06445 |
| 2023 | 0.06992 |
| 2024 | 0.06656 |
| 2025 | 0.05780 |

CV descriptivo **≈7.4 %**; mejor año **2025**, peor **2023**. Por estación, menor RMSE **IB08=0.05462** y mayor **IB11=0.08176 mm/día**. IB08 puede tener evaluación histórica válida aunque su Forecast actual esté temporalmente indisponible: son componentes distintos.

La mejora frente a AHC aparece en los **cinco años**, **nueve estaciones**, **cuatro temporadas climáticas**, **doce meses** y **45 grupos estación-año** revisados. No significa idéntico error en todos los grupos ni garantía para periodos futuros.

Entre las **35 configuraciones**, se conservan **1.007 predicciones negativas**, todas de Linear, Ridge o Elastic Net. **No hay clipping.** Son comportamiento/limitación de esas familias y deben permanecer visibles, sin ocultarlas ni borrar filas para mejorar métricas. El champion no produjo negativas ni valores no finitos.

## Recuperación de TEST: alcance honesto del cierre

La selección y los modelos DEVELOPMENT estaban congelados antes de cualquier acceso de recuperación a TEST. El intento original quedó **TEST_ACCESS_UNKNOWN**. El [registro de incidente](scientific_pipeline/docs/ml-v1-test-recovery-access-incident.md) conserva además el primer intento Kaggle como TEST_ACCESS_OCCURRED / TEST_EVALUATION_NOT_PERFORMED: no produjo métricas y su manifest no describía completamente el acceso técnico.

La recuperación final se etiqueta **retrospective_recovery_evaluation** y evalúa los mismos modelos serializados. **No hubo reselección, entrenamiento ni refit después de TEST.** El cierre PASS no borra los intentos ni convierte el periodo en pristine/prospective holdout. No afirmar que TEST se abrió por primera vez o una sola vez a lo largo de todo el proceso.

La ausencia de uso de TEST en las decisiones de ajuste/selección y la ausencia de un holdout prístino son afirmaciones distintas. Ambas se documentan. Una confirmación prospectiva exigiría nuevos datos y protocolo previo.

## Reproducibilidad, ubicación y preservación

Identidades comunicadas del cierre autorizado, coherentes con la preservación registrada en main:

| Elemento | Valor |
|---|---|
| Run | `ml-recovery-c2c5c34e7cc642a181d0f3ee12a5ef7e` |
| Pipeline | `94ebf67df2954e6818fc5f70c6b2e91b1b07f505` |
| Dataset SHA256 | `0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb` |
| Protocol SHA256 | `45f1c4f3ed20bb70dc3b1ca06a8e23ec14f95aae6b60f574e44a8a261cc07048` |
| Run manifest SHA256 | `05834edaebf3f990729ebb8b15ecd7511bb96e006bfab69f11408a250f432aab` |
| Artefacto canónico | `/srv/siar-project/scientific-artifacts/ml-v1-test-recovery-20261101T090828Z/` |
| Receipt | `science-ml-v1-test-recovery-20261101T090828Z_83cf4565.json` |
| Snapshot | `backup_20261101T095440Z_0381c133` |
| Verificación remota | REMOTE_VERIFIED |
| Restore | 77 archivos, RESTORE_HASHES_PASS |

El receipt se localiza bajo `/srv/siar-project/ingestion-receipts/`. Esta misión no vuelve a ejecutar el run, acceder al servidor, restaurar sus bytes ni modificar artefactos. Git contiene documentación/evidencia versionable, no necesariamente el conjunto masivo de predicciones/modelos. [Localizaciones y linaje adicional](docs/DATA_LOCATIONS.md), [metodología técnica](scientific_pipeline/docs/ml_v1_classical_methodology.md), [operación de preservación](docs/backup/README.md).

## Estado posterior

Dataset v1 **PASS** → cuatro baselines **PASS** → AHC **PASS** → ML clásico **PASS**. **ANN moderna NOT STARTED**. **WEB-V0.3-ML READY** para consumir ciencia publicada, no desplegada por esta misión. La web consulta resultados sin volver a entrenar ni calcular ciencia.

La [bitácora 1.2](docs_1.2_Tratamiento%20de%20datos%20y%20c%C3%A1lculo%20de%20estimaciones.md) conserva el tratamiento histórico; [PROJECT_STATUS](docs/PROJECT_STATUS.md) expresa el estado rápido y [ROADMAP](docs/ROADMAP.md) fija los siguientes gates.
