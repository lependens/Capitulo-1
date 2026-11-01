> **Cierre posterior — 01/11/2026: ML-V1-CLASSICAL-MODELS PASS**, autorizado por Dirección 00. Dataset v1, cuatro baselines y AHC PASS; ANN moderna NOT STARTED; WEB-V0.3-ML READY. Los apartados previos de preparación/TEST pendiente se conservan como protocolo o historia, no como estado actual. La ejecución final fue **retrospective_recovery_evaluation**: selección y modelos DEVELOPMENT congelados antes de recovery, intento original TEST_ACCESS_UNKNOWN, sin reselección/entrenamiento/refit después de TEST. Las menciones a una evaluación única describen el diseño inicial, no el historial real; no afirmar holdout prístino/prospectivo. [Resultados, interpretación y linaje](../../docs_1.2.6_Modelos%20ML%20cl%C3%A1sicos%20para%20estimaci%C3%B3n%20de%20ET0.md).

> **Estado posterior — 31/10/2026:** WEB-V0.2-AHC ya está desplegada (PR #46/#47) y el runtime ML rolling está implementado (PR #48); rolling validation y Linux smoke PASS. La campaña científica oficial permanece pendiente de despliegue/ejecución; ANN NOT STARTED. Las referencias inferiores a «web no construida», ML futuro o merge pendiente conservan el estado previo de su entrega. No cambian la semántica contractual ni resultados congelados. Para el contrato rolling vigente, véase [refinamiento temporal](ml_validation_contract_refinement.md); estado rápido en [PROJECT_STATUS](../../docs/PROJECT_STATUS.md).

# Contrato experimental para ML — v1

**Estado documental vigente — 31/10/2026: protocolo ML-V1 congelado; runtime rolling implementado, campaña oficial pendiente de ejecución tras merge.** AHC temporal ya está PASS. No hay campaña ML oficial ejecutada ni ANN entrenada; no hay web construida. El runtime de campaña se describe en [refinamiento temporal](ml_validation_contract_refinement.md). [Estado maestro](../../docs/PROJECT_STATUS.md).

Este contrato define el experimento ML v1; **no declara modelos entrenados ni
resultados ML oficiales**. El cierre AHC es PASS; la campaña ML sigue pendiente de
merge y ejecución. El protocolo
legible por máquina es [ml-v1-contract.json](../experiments/ml-v1-contract.json).
La metodología AHC vigente se conserva en [ahc_methodology.md](ahc_methodology.md).
ANN y la construcción de una interfaz web quedan fuera de esta fase.

## Pregunta, referencias e identidad

La tarea primaria será emular **PM_FAO56_V1**, una estimación calculada de ET₀,
en mm/día. EtPMon será una referencia calculada SIAR secundaria, nunca una
feature. Ninguna representa por sí misma ET medida. Con todos los inputs de PM,
un modelo aproxima una ecuación conocida: su rendimiento mide emulación y
disponibilidad de información, no una validación física independiente.

Dataset Observational v1 queda congelado con SHA-256
`0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb`.
Cada ejecución futura tendrá experiment_id propio y conservará dataset_id/hash,
audit_source_repo_sha, source_repo_sha, POST ID/hash, productores de predicciones,
pipeline SHA, configuración/hash, entorno, semillas, timestamps UTC y hashes de
salidas. Cambiar target, variables, reglas, cortes o hiperparámetros crea otra
configuración identificada; no sobrescribir resultados ni inventar un SHA de
pipeline antes de ejecutar. El hash del manifest se publica fuera del propio
documento para evitar autorreferencia.

## Partición y alcance de la validación

| Ámbito | Fechas inclusivas globales |
|---|---|
| DEVELOPMENT rolling-origin | 2004-01-01–2020-12-31 |
| TEST | 2021-01-01–2025-12-31 |

DEVELOPMENT usa F1 2004–2009→2010–2012, F2 2004–2012→2013–2015,
F3 2004–2015→2016–2018 y F4 2004–2018→2019–2020. La misma fecha pertenece
al mismo fold en todas las estaciones. Publicar un archivo de asignación de
claves estación-fecha y su hash. No mover cortes según errores, target o
cobertura individual. El TRAIN de cada fold ajusta modelo y transformaciones;
las ventanas futuras agregadas seleccionan. La configuración congelada se
reajusta en DEVELOPMENT completo y TEST evalúa una vez.
Presupuesto de candidatos, hiperparámetros, semillas y criterio de selección se
deben cerrar en un protocolo de ejecución antes de ajustar. Este documento no
autoriza buscar configuraciones reiteradamente contra TEST.

Los baselines históricos completos ya se examinaron y el cierre AHC también
puede haber publicado el periodo TEST. Por ello, el estado es
**retrospective_locked**: separación temporal del ajuste, sin afirmar que TEST
sea desconocido. Comparar con AHC no restaura esa independencia. Una confirmación
prospectiva necesita un periodo posterior, dataset nuevo y protocolo previo.

Para el benchmark AHC, OLS0=Σ(x·PM)/Σ(x²), ajustado solo en TRAIN, permanece
congelado sin refit TRAIN+VALIDATION. ML no modifica esa referencia de comparación.
Si se propusiera un refit operativo posterior, sería otro experimento sin
atribuirle las métricas del modelo congelado.

## Familias de información

Todas las familias comparten el contexto explícito
`latitude_deg, altitude_m, doy_sin, doy_cos, Ra_mj_m2_day`.
Los términos cíclicos usan el número real de días del año; Ra se calcula por la
geometría solar versionada. No se deriva información de los targets. El contexto
se incluye en cada familia, y el nombre completo lleva sufijo `_WITH_CONTEXT_V1`;
no confundir estos experimentos con modelos que solo reciben temperatura.

Se utilizan Tcalc=(Tmax+Tmin)/2 y TD=Tmax−Tmin, evitando la dependencia lineal
exacta de incluir a la vez Tmin, Tmax y Tcalc en regresión. TempMedia SIAR sigue
preservada en Dataset v1, pero no sustituye Tcalc ni se añade implícitamente.

| Familia | Variables meteorológicas además del contexto |
|---|---|
| FEATURE_SET_T | Tcalc, TD |
| FEATURE_SET_T_RH | Tcalc, TD, HumedadMedia, HumedadMin, HumedadMax |
| FEATURE_SET_T_RS | Tcalc, TD, Radiacion |
| FEATURE_SET_T_RH_RS | Tcalc, TD, HumedadMedia, HumedadMin, HumedadMax, Radiacion |
| FEATURE_SET_FULL | Tcalc, TD, HumedadMedia, HumedadMin, HumedadMax, Radiacion, VelViento |

Presión puede derivarse de altitud dentro de una transformación versionada;
no añadirla junto con altitud como predictor distinto sin declararlo. FULL no
significa todas las columnas raw. Precipitación, suelo, dirección y horas de
extremos necesitan hipótesis y versión posteriores. Las unidades son las
aceptadas por v1: °C, %, MJ/m²/día, m/s, m y grados; sin reconversiones de Rs/viento.

No se admite como predictor EtPMon, PePMon, PM calculado, ningún baseline,
salida/coeficiente AHC, ratio de calibración, residuo o flag construida usando
target/error. station_code es una clave, no una variable ordinal numérica.
Cualquier efecto por estación debe ser una variante explícita; no demuestra
transferencia geográfica. No introducir filtros aprendidos con todo el histórico,
interpolación bidireccional, ventanas centradas ni información futura. Lags,
horizontes forecast y embargo quedan pendientes de otro contrato causal.

## Elegibilidad, missing y estaciones

Conservar Quality PASS/WARN y observaciones OBSERVED/RECOVERED oficiales, sin
modificar QC o fuentes. No imputar el target. Elegibilidad de predicción, target
PM disponible y referencia SIAR disponible son indicadores diferentes. La regla
primaria de evaluación exige PM finito/no negativo; EtPMon secundaria conserva
la política v1, incluyendo ceros. No eliminar predicciones negativas a posteriori
ni recortarlas silenciosamente: conservar y marcar como diagnóstico.

Primera política: casos completos por familia, con variables dentro del dominio
científico registrado, sin imputación meteorológica. Una ausencia no elimina la
fila de otras familias ni del dataset. Publicar conteos por causa de exclusión.
Una variante futura con imputación debe ajustar parámetros solo en TRAIN y
compararse separadamente con indicadores de ausencia declarados.

La población primaria comprende estaciones con soporte TRAIN y presencia en
VALIDATION/TEST; congelar la lista a partir de disponibilidad antes de ajustar.
IB09 mantiene Quality FAIL y queda excluida. IB07, cerrada en 2012, no tiene
validación/test. IB101, serie corta 2008–2009, no se convierte en estación robusta
por imputación ni recibe fallback AHC. Las nueve activas con las tres particiones
constituyen la población candidata, sujeta a elegibilidad por familia. Si una
estación no estuvo en TRAIN, su evaluación pertenece a transferencia geográfica
separada; no mezclarla con generalización temporal en estaciones conocidas.

Para regresiones, medias/desviaciones u otra transformación se estiman en TRAIN
y se almacenan. Varianza cero se resuelve por política registrada, sin dividir
por cero. Árboles pueden omitir escalado. No normalizar cada año de test con sus
propias estadísticas ni ajustar valores extremos mediante residuos futuros.

## Selección, comparación y métricas

La campaña congelada cubre regresión lineal, Ridge/Elastic Net, árbol limitado,
random forest, extra trees y boosting sobre cinco familias de inputs; el grid
contiene 185 candidatos y selecciona 35 celdas finales por el protocolo
rolling-origin. La regla exacta y los desempates están congelados en
[`ml-v1-classical.json`](../experiments/ml-v1-classical.json) y en
[`ml_v1_classical_methodology.md`](ml_v1_classical_methodology.md). ANN queda
expresamente pospuesta hasta analizar estabilidad, cobertura y ganancias fuera
de muestra.

Publicar cohortes propias para disponibilidad y cohortes comunes con hash de
claves para accuracy comparable. Las comparaciones entre familias no deben
atribuir a una nueva variable un cambio de error causado por cambiar de filas.
La tabla común de referencia incluye PM, los tres empíricos originales y sus
tres ajustes anuales AHC_VALIDATED; variantes estacionales/mensuales seleccionadas
solo en VALIDATION permanecen secundarias y con identidad distinta. Nunca PM
ajustado ni AHC_CLIMATOLOGY en una comparación fuera de muestra.

Métricas por TRAIN/VALIDATION/TEST: n, MAE, RMSE, bias=predicción−referencia y
R²=1−SSE/SST; MAE/RMSE/bias en mm/día. Referencias cero permanecen. R² es null
si n<2 o SST=0, con motivo. El agregado global se calcula agrupando parejas,
sin promediar RMSE/R² de estaciones. Publicar también estación, año, mes y
temporada; DJF asigna diciembre al año siguiente. TRAIN es ajuste dentro de
muestra; TEST es la métrica principal de generalización temporal retrospectiva.
Informar precisión junto a inputs requeridos, cobertura y limitaciones.
