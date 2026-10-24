# AHC — metodología y validación temporal oficial

AHC-METHODOLOGY-AND-TEMPORAL-VALIDATION = PASS.
Código: PR #38, pipeline `38503dbe8422982deeb3680bbc3e82fd950da4e1`.
Dataset v1: `0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb`.
Raw, Quality y Dataset intactos; cuatro baselines importados sin recalcular ni modificar.

## Pregunta y protocolo

Multiplicador adimensional por estación a=Σ(x·PM)/Σ(x²), OLS sin intercepto.
Referencia primaria PM_FAO56_V1; EtPMon, calculada SIAR, secundaria.
Ninguna es ET medida. Para HS, el coeficiente físico es 0.0023·a.
Media de cocientes y cociente de sumas son sensibilidades TRAIN, no estimadores elegidos por TEST.
[Bibliografía y decisiones](../../docs/ahc_methodology.md), [protocolo](../../experiments/ahc-v1.json).

TRAIN 2004–2015; VALIDATION 2016–2020; TEST 2021–2025. Cortes globales,
sin random split ni refit TRAIN+VALIDATION. TEST **retrospective_locked**:
baselines de todo el histórico ya inspeccionados. Ajuste/selección AHC preceden
TEST; confirmación prospectiva requiere datos futuros y otro experimento.

## Métricas TEST principales — cohorte común de siete, referencia PM

PM contra sí misma es control de identidad, no evidencia de exactitud física.
Las siete filas usan las mismas claves; MAE/RMSE/bias en mm/día, R²=1−SSE/SST.

| method_id | n | mae | rmse | bias | r2 |
| --- | --- | --- | --- | --- | --- |
| PM_FAO56_V1 | 15958 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| HS_RA_V1 | 15958 | 0.786298 | 0.948651 | 0.571058 | 0.692346 |
| HG_RS_V1 | 15958 | 0.471068 | 0.564675 | 0.287469 | 0.890995 |
| VALIANTZAS_V1 | 15958 | 0.474368 | 0.628676 | 0.242865 | 0.864885 |
| HS_RA_AHC_VALIDATED | 15958 | 0.504081 | 0.638838 | 0.230566 | 0.860481 |
| HG_RS_AHC_VALIDATED | 15958 | 0.318930 | 0.414241 | 0.003436 | 0.941338 |
| VALIANTZAS_AHC_VALIDATED | 15958 | 0.392259 | 0.523955 | 0.166835 | 0.906149 |

## Métricas TEST — cohortes propias, referencia PM

El n propio puede diferir: no comparar diferencias de error como efectos causales.

| method_id | n | mae | rmse | bias | r2 |
| --- | --- | --- | --- | --- | --- |
| HG_RS_AHC_VALIDATED | 15958 | 0.318930 | 0.414241 | 0.003436 | 0.941338 |
| HG_RS_AHC_VALIDATED_SELECTED | 15958 | 0.306919 | 0.401392 | -0.011504 | 0.944921 |
| HG_RS_V1 | 15958 | 0.471068 | 0.564675 | 0.287469 | 0.890995 |
| HS_RA_AHC_VALIDATED | 15958 | 0.504081 | 0.638838 | 0.230566 | 0.860481 |
| HS_RA_V1 | 15958 | 0.786298 | 0.948651 | 0.571058 | 0.692346 |
| PM_FAO56_V1 | 15958 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| VALIANTZAS_AHC_VALIDATED | 15958 | 0.392259 | 0.523955 | 0.166835 | 0.906149 |
| VALIANTZAS_V1 | 15958 | 0.474368 | 0.628676 | 0.242865 | 0.864885 |

## Referencia calculada SIAR — TEST común

| method_id | n | mae | rmse | bias | r2 |
| --- | --- | --- | --- | --- | --- |
| PM_FAO56_V1 | 15952 | 0.011214 | 0.023774 | -0.000759 | 0.999807 |
| HS_RA_V1 | 15952 | 0.785742 | 0.949108 | 0.570193 | 0.693142 |
| HG_RS_V1 | 15952 | 0.470076 | 0.563848 | 0.286601 | 0.891699 |
| VALIANTZAS_V1 | 15952 | 0.475139 | 0.629684 | 0.242029 | 0.864932 |
| HS_RA_AHC_VALIDATED | 15952 | 0.504902 | 0.640096 | 0.229760 | 0.860428 |
| HG_RS_AHC_VALIDATED | 15952 | 0.318809 | 0.414792 | 0.002644 | 0.941391 |
| VALIANTZAS_AHC_VALIDATED | 15952 | 0.393344 | 0.524913 | 0.166054 | 0.906140 |

## Complejidad seleccionada antes de TEST

| method | granularity | validation_n | annual_validation_rmse |
| --- | --- | --- | --- |
| HS_RA_V1 | annual | 15900 | 0.638105 |
| HG_RS_V1 | seasonal | 15900 | 0.416380 |
| VALIANTZAS_V1 | annual | 15900 | 0.499820 |

HS estacional/mensual no alcanza la mejora mínima de RMSE VALIDATION del 2 %.
HG-Rs estacional cumple; mensual no mejora otro 2 % respecto a estacional.
Valiantzas no activa complejidad por materialidad TRAIN. No se reelige tras TEST.
La comparación seleccionada HG-Rs es secundaria y distinta del benchmark anual:

| method_id | n | mae | rmse | bias | r2 |
| --- | --- | --- | --- | --- | --- |
| HG_RS_V1 | 15958 | 0.471068 | 0.564675 | 0.287469 | 0.890995 |
| HG_RS_AHC_VALIDATED | 15958 | 0.318930 | 0.414241 | 0.003436 | 0.941338 |
| HG_RS_AHC_VALIDATED_SELECTED | 15958 | 0.306919 | 0.401392 | -0.011504 | 0.944921 |

## Estabilidad y coeficientes

La sensibilidad de referencia sobre parejas TRAIN robustas da una diferencia
máxima de coeficiente PM/EtPMon de **0,258 %** y de predicción ajustada media
de **0,00752 mm/día**. No justifica cambiar la referencia predeclarada.
Los estimandos sí difieren: mean-ratio llega a separarse **29,96 %** de OLS0,
y ratio-sums **4,19 %**. No son implementaciones equivalentes del mismo ajuste.
Se conserva OLS0 por la pregunta diaria de mínimos cuadrados fijada antes de
calibrar, sustentada por la regresión por origen de Martí; no por competir en TEST.
Estos resultados tampoco reproducen el AHC legacy de histórico completo.

CV muestral máximo entre bloques completos robustos: 6.612 %; max/min máximo: 1.164415. Ningún gate de STOP se activa.

TRAIN usa bloques 2004–2009/2010–2015 para decisiones. Después de TEST,
2004–2010/2011–2015/2016–2020/2021–2025 describen variación y tendencia
Theil–Sen; pocos bloques no sustentan inferencia de significación ni causalidad.
[Diagnósticos por bloque](closeout/closeout.json) incluyen n, soporte, media,
desviación muestral, rango, CV y tendencia. IB07 tiene cobertura posterior limitada.

| station | method | TRAIN_a | TRAIN_n | CLIMATOLOGY_a | CLIMATOLOGY_status |
| --- | --- | --- | --- | --- | --- |
| IB01 | HS_RA_V1 | 0.902147 | 4146 | 0.877452 | ROBUST |
| IB01 | HG_RS_V1 | 0.922581 | 4146 | 0.921040 | ROBUST |
| IB01 | VALIANTZAS_V1 | 0.985506 | 4146 | 0.963672 | ROBUST |
| IB02 | HS_RA_V1 | 0.900268 | 4218 | 0.887901 | ROBUST |
| IB02 | HG_RS_V1 | 0.946427 | 4218 | 0.950553 | ROBUST |
| IB02 | VALIANTZAS_V1 | 0.982224 | 4218 | 0.973595 | ROBUST |
| IB03 | HS_RA_V1 | 0.906438 | 4219 | 0.887869 | ROBUST |
| IB03 | HG_RS_V1 | 0.946184 | 4219 | 0.949717 | ROBUST |
| IB03 | VALIANTZAS_V1 | 0.989581 | 4219 | 0.979389 | ROBUST |
| IB04 | HS_RA_V1 | 0.870862 | 4219 | 0.857624 | ROBUST |
| IB04 | HG_RS_V1 | 0.903607 | 4219 | 0.897300 | ROBUST |
| IB04 | VALIANTZAS_V1 | 0.934908 | 4219 | 0.925179 | ROBUST |
| IB05 | HS_RA_V1 | 0.854832 | 4216 | 0.820904 | ROBUST |
| IB05 | HG_RS_V1 | 0.882419 | 4216 | 0.871289 | ROBUST |
| IB05 | VALIANTZAS_V1 | 0.913790 | 4216 | 0.887353 | ROBUST |
| IB06 | HS_RA_V1 | 0.818645 | 4209 | 0.794537 | ROBUST |
| IB06 | HG_RS_V1 | 0.917002 | 4209 | 0.912039 | ROBUST |
| IB06 | VALIANTZAS_V1 | 0.931762 | 4209 | 0.912461 | ROBUST |
| IB07 | HS_RA_V1 | 0.756686 | 2929 | 0.756686 | ROBUST |
| IB07 | HG_RS_V1 | 0.819466 | 2929 | 0.819466 | ROBUST |
| IB07 | VALIANTZAS_V1 | 0.795687 | 2929 | 0.795687 | ROBUST |
| IB08 | HS_RA_V1 | 0.874687 | 3614 | 0.832995 | ROBUST |
| IB08 | HG_RS_V1 | 0.876619 | 3614 | 0.861656 | ROBUST |
| IB08 | VALIANTZAS_V1 | 0.931350 | 3614 | 0.897392 | ROBUST |
| IB09 | HS_RA_V1 | null | 0 | null | UNAVAILABLE |
| IB09 | HG_RS_V1 | null | 0 | null | UNAVAILABLE |
| IB09 | VALIANTZAS_V1 | null | 0 | null | UNAVAILABLE |
| IB10 | HS_RA_V1 | 0.831501 | 3649 | 0.800392 | ROBUST |
| IB10 | HG_RS_V1 | 0.886143 | 3649 | 0.862090 | ROBUST |
| IB10 | VALIANTZAS_V1 | 0.909319 | 3649 | 0.879483 | ROBUST |
| IB101 | HS_RA_V1 | null | 246 | 1.096818 | SHORT_SERIES_DESCRIPTIVE |
| IB101 | HG_RS_V1 | null | 246 | 0.903114 | SHORT_SERIES_DESCRIPTIVE |
| IB101 | VALIANTZAS_V1 | null | 246 | 1.130896 | SHORT_SERIES_DESCRIPTIVE |
| IB11 | HS_RA_V1 | 1.213275 | 3373 | 1.150818 | ROBUST |
| IB11 | HG_RS_V1 | 0.910731 | 3373 | 0.912962 | ROBUST |
| IB11 | VALIANTZAS_V1 | 1.219688 | 3372 | 1.178904 | ROBUST |

AHC_CLIMATOLOGY se calcula solo después de completar TEST, usando todo el
histórico declarado hasta 2025. No sustituye al coeficiente TRAIN ni acredita
rendimiento fuera de muestra. IB101: cifra climatológica descriptiva de serie
corta, deployable=false; coeficiente validado null. IB09: Quality FAIL, sin ajuste.
IB07: ajuste TRAIN con soporte, sin validación/test posteriores al cierre.
No se transfieren coeficientes regionales ni de otras estaciones.

[Cambios por estación TEST](station_test_changes.json) conservan también
empeoramientos; no se retocan coeficientes para corregirlos. Un mejor resultado
agrupado no garantiza mejora en todas las estaciones o periodos. HS requiere
temperaturas/geometría; VAL añade HR; HG-Rs añade Rs; PM integra viento,
humedad, Rs y presión/altitud. Las tablas de cobertura/complejidad permiten
evaluar coste informativo y disponibilidad junto a error, sin afirmar causalidad.

La mejora RMSE anual agrupada se observa en las nueve estaciones TEST para
los tres ajustes, pero no en cada estación-año. [Revisión temporal posterior](period_review.json)
registra ocho empeoramientos entre los grupos con n≥300. Destaca HS en IB11/2025:
RMSE 0,756→0,897 (+18,57 %, n=365). IB11/2023 presenta los mayores RMSE
estación-año ajustados: HS 0,929; HG-Rs 0,684; VAL 0,827. Son diagnósticos
retenidos, sin atribuir causas ni volver a seleccionar modelos. La estabilidad
del multiplicador no garantiza estabilidad del error o ausencia de bias.

## Artefactos, contratos y verificación

[Índice completo de archivos, tamaños, rutas y SHA256](artifact-index.json).
CSV diarios, métricas extensas y asignaciones temporales permanecen fuera de Git;
los manifests nativos y evidencia compacta se preservan aquí sin cambiar bytes.
Las rutas absolutas de manifests identifican el productor local; para trasladar
archivos, conservar hashes y documentar resolución, sin reescribir manifests.
[Verificación](verification.json): fuentes/POST/catálogo/dataset y todos los
outputs por hash; 75.496 claves diarias; coeficientes congelados; métricas TEST
comunes recomputadas desde el CSV. Filas exteriores y ceros SIAR preservados.
La proyección diaria conserva 50 ceros EtPMon, incluidos los no evaluables por
Quality; la elegibilidad original determina las parejas de métricas.

[Reproducción en destino nuevo](reproduce.ps1) requiere checkout limpio del
pipeline publicado y rutas al dataset y manifest de cuatro baselines preservados.
No necesita red, adquisición SIAR ni recalcular baselines. Cada etapa comprueba
identidades/hashes y el éxito del padre antes de continuar.

[Contrato de visualización](../../docs/visualization_contract.md): daily_predictions,
station/monthly/seasonal/yearly_metrics, model/station_metadata, cobertura,
complejidad y esquemas reales. Web no construida; el consumidor no recalcula ciencia.
[Contrato ML](../../docs/ml_experiment_contract.md): target, familias de inputs,
splits/cohortes, missing, normalización TRAIN y leakage. Ningún modelo ML entrenado;
ANN queda posterior. Reutilizar este TEST para decisiones futuras no crea un
test nunca visto: su condición retrospectiva debe acompañar cada resultado.

Limitaciones retenidas: referencias calculadas, altura viento SIAR convencional
2 m sin certificación exhaustiva estación-periodo, provenance parcial, IB09 FAIL,
IB101 corto, ausencia de test futuro IB07, incertidumbre de coeficientes no
cuantificada y umbrales pragmáticos predeclarados. Artefactos masivos locales:
backup independiente todavía no certificado (PROJECT-ARTIFACT-BACKUP-V1 pendiente).
