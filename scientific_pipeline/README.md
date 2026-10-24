# SIAR Science — ET₀ baselines v1

Capa reproducible de observaciones SIAR Baleares: validación científica por método, Penman–Monteith diario FAO-56, Hargreaves–Samani fijo, VALIANTZAS_V1, Hargreaves-Rs y evaluación frente a EtPMon. Conserva valores originales y genera resultados derivados separados. No incorpora forecast, ML, AHC ni limpieza global ±3σ.

La forense queda **RESOLVED_FOR_V1** por decisión de Dirección 00 del 2026-10-19. Radiacion se interpreta directamente en MJ m⁻² día⁻¹ y VelViento en m/s, con convención SIAR de viento a 2 m. `UNITS_PROVENANCE_PARTIAL` y `station_period_provenance_not_exhaustively_verified` son limitaciones registradas, no bloqueos. Una evidencia positiva de altura distinta de 2 m requiere resolver el cálculo PM afectado.

**Estado vigente — 2026-10-24:** Dataset Observational v1 construido/validado desde POST corregido SIAR-QA-POST-20261020T193140Z y handoff autorizado. Comparación oficial PM_FAO56_V1 / HS_RA_V1 / VALIANTZAS_V1 / HG_RS_V1 **PASS**, con CSV verificado y 75 tests/CI Python 3.11/3.12/3.13 PASS. [Resultados de cuatro métodos](results/four-classical-20261024T042250Z/README.md). Se conserva la [comparación previa de tres](results/classical-20261024T034529Z/README.md). Se preserva la [primera ejecución PM/HS](results/pm-vpd-20261023T194553Z/README.md), publicada por PR #31 tras el fix PR #30. READY_FOR_DATASET_V1 y sus tests anteriores son hitos históricos; ver [estado de misión](docs/mission_status.md).

## Instalación y ayuda

Python 3.11 o posterior. Desde la raíz del repositorio, en un entorno Python de trabajo:

```powershell
python -m pip install -e ./scientific_pipeline
python -m siar_science --help
```

No se necesitan credenciales SIAR ni acceso a la API para ejecutar la capa sobre entradas locales autorizadas. La interfaz está implementada y dispone de ayuda por comando con `--help`.

## Interfaz y reproducción

Los siguientes ejemplos usan rutas ilustrativas; no son las rutas reales del dataset y run ya producidos. Para reproducir, utilizar el handoff autorizado, POST corregido, blobs fuente y hashes identificados en el informe oficial; crear destinos nuevos sin sobrescritura. Esta documentación no lanza una nueva ejecución.

```powershell
python -m siar_science build-dataset --handoff ./handoff/quality-closeout.json --output ./derived/observational-v1.json
python -m siar_science validate --dataset ./derived/observational-v1.json --output ./runs/v1-validation
python -m siar_science run-baselines --dataset ./derived/observational-v1.json --output ./runs/v1-baselines
```

`build-dataset` verifica el gate `SIAR-QUALITY-CLOSEOUT`, autorización y hashes, los tres informes POST y las fuentes; genera un dataset nuevo. No descarga, repara o reescribe observaciones. El [contrato de handoff](docs/contracts.md) fija rutas, campos y correspondencia con los esquemas oficiales Quality.

El handoff distingue dos SHA obligatorios: `audit_source_repo_sha` identifica el
commit realmente auditado por POST y `source_repo_sha` la publicación canónica
de los CSV, catálogo y POST. Pueden ser diferentes: los informes conservan su
`repo_git_sha`, validado contra el primero, y la continuidad con el segundo se
comprueba mediante los hashes de todos los archivos y del catálogo. El segundo
no es el commit del propio handoff; este puede publicarse después, apuntando a
una publicación canónica que ya contiene los artefactos. No se infiere el SHA
auditado si falta y no se reescribe POST para hacerlos coincidir.

`validate` verifica integridad y elegibilidad científica sin producir un valor ET₀ ficticio. `run-baselines` calcula ambos métodos y sus evaluaciones propias y pareadas en una ejecución. También se ofrecen etapas independientes:

```powershell
python -m siar_science calculate-pm --dataset ./derived/observational-v1.json --output ./runs/v1-pm
python -m siar_science calculate-hargreaves --dataset ./derived/observational-v1.json --output ./runs/v1-hs
python -m siar_science evaluate --dataset ./derived/observational-v1.json --predictions ./runs/v1-pm/predictions.json --predictions ./runs/v1-hs/predictions.json --output ./runs/v1-evaluation
```

`evaluate` consume predicciones ya calculadas: `--predictions` es obligatorio y repetible. Cada archivo debe corresponder al mismo `dataset_id` y hash de bytes del dataset; los identificadores de método no pueden solaparse. Puede recibir el archivo combinado de `run-baselines` en lugar de los dos archivos individuales. No vuelve a calcular modelos para sustituir silenciosamente sus entradas.

Los artefactos de predicción conservan la identidad del productor y su configuración. La evaluación verifica el estado exitoso y los parámetros versionados, conservando el origen de cada archivo aunque proceda de otra revisión compatible. Con un solo modelo, su cohorte propia sigue disponible y la cohorte pareada PM/HS tiene n=0.

Dataset, manifiestos y predicciones conservan ambos SHA de procedencia; las
métricas serializadas incluyen también ambos campos. La evaluación rechaza
predicciones cuyos SHA auditado/canónico no correspondan al dataset. Esta
separación no cambia PM, HS, QC, elegibilidad, unidades ni fórmulas de métricas.

Los comandos con `--dataset` admiten opcionalmente `--pipeline-sha` con el SHA exacto de la implementación revisada. Sin esa opción se identifica el checkout disponible. No usar un identificador inventado ni interpretar un SHA declarado como certificado de árbol limpio.

## Resultados y trazabilidad

El JSON completo del dataset y predictions.json masivo se preservan localmente; Git contiene evidencia compacta, metrics.json y manifest.json. [Dónde consultar datos/resultados](../docs/DATA_LOCATIONS.md). Dataset SHA256: `0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb`.

Cada comando de ejecución crea un directorio nuevo con:

| Archivo | Contenido |
|---|---|
| `predictions.json` | Resultados por estación/fecha/método, elegibilidad, motivos y flags; `null` cuando no se calcula |
| `metrics.json` | Cohortes propias y pareadas, tamaños, métricas y agrupaciones; valores no definidos como `null` |
| `manifest.json` | Identidades/hashes de entrada y outputs, versión/SHA del pipeline, configuración, políticas, limitaciones y estado de ejecución |

Validación conserva las filas y sus metadatos de elegibilidad, con ET₀ sin calcular y la colección de métricas vacía. No se permite sobrescribir una ejecución anterior. Se preservan EtPMon y TempMedia SIAR, flags de origen y referencias POST. Una ejecución sintética verifica software, no crea Dataset v1 ni certifica observaciones reales.

## Decisiones científicas

Los cuatro baselines usan Tcalc=(Tmax+Tmin)/2, conservando TempMedia original. PM fija G=0, γ=0.000665P y albedo=0.23; calcula presiones de vapor con extremos térmicos y de humedad. La nubosidad de Rnl limita Rs/Rso por arriba a 1 y **no aplica floor inferior por defecto**. Los ratios extremos, Rnl negativo y ET₀ negativa se marcan; no se recorta el resultado a cero.

HS usa el coeficiente fijo **0.0023·0.408**, Ra calculada, temperatura y rango térmico. No usa Rs medida ni coeficientes ajustados. La elegibilidad depende de sus propias entradas: humedad o radiación medidas ausentes pueden impedir PM y permitir HS; EtPMon ausente impide evaluación y permite predicción.

Error=predicción−EtPMon. Se publican n, MAE, RMSE, MBE/bias y R² predictivo (1−SSE/SST), conservando los ceros de referencia. La referencia evaluable es finita, no negativa y procede de una observación aceptada dentro de periodo oficial; los valores negativos se preservan con flag y se excluyen de métricas. Se distinguen cohortes propias de cada modelo y la intersección pareada. EtPMon es una estimación SIAR de comparación, no verdad independiente de terreno. No se presentan las métricas históricas del legado como resultados de esta nueva implementación.

Los cuatro baselines oficiales no aprenden parámetros. El AHC legacy full-series/in-sample no es validación predictiva. AHC-METHODOLOGY-REVIEW y el contrato temporal se resolverán en misión separada antes de AHC_V1: se distinguirán usos climatológicos y calibración predictiva, estabilidad temporal y posibles escalas anual/estacional/mensual. Si AHC se estima de datos es parámetro calibrado; esta PR no decide todo el dataset frente a solo TRAIN. La evaluación predictiva futura debe prevenir leakage y ML mantiene separación temporal y ajuste de transformaciones sin usar datos de evaluación.

## Referencias de la entrega

- [HG_RS_V1: cuarto baseline con Rs medida, contrato y unidades](docs/hargreaves_rs_v1.md).

- [VALIANTZAS_V1: fórmula adoptada y revisión de unidades](docs/valiantzas_v1.md).

- [Primera ejecución oficial PM/HS: resultados, cohortes, hashes y limitaciones](results/pm-vpd-20261023T194553Z/README.md).
- [Supuestos y diferencias con el legado](docs/assumptions.md).
- [Contrato de dataset, handoff y ejecución](docs/contracts.md).
- [Estado de misión y condiciones pendientes](docs/mission_status.md).
- [Auditoría del legado](docs/legacy_audit.md).
- [Evidencia offline de unidades](docs/units_offline_evidence.md).
- [Resultados numéricos forenses preservados](docs/units_offline_forensic.md), cuyo diagnóstico preliminar se interpreta con la decisión posterior de 00.

Pruebas locales: `python -m unittest discover -s scientific_pipeline/tests -v`. La suite matemática y sintética verifica fórmulas publicadas, QC/eligibilidad, métricas, handoff/POST/catálogo, hashes, protección de outputs, procedencia y CLI. La CI científica ejecuta el paquete instalado y la CLI en Python 3.11, 3.12 y 3.13. El cierre y las evidencias verificadas se registran en el estado de misión.

## Tercer baseline clásico y exportación

`VALIANTZAS_V1` adopta Martí et al. (2024), Ecs.1b+3, con factor energético fijo
0.408: coeficiente expandido **0.001861704**, factor de humedad
**(1.001−HumedadMedia/100)^0.2**, Tcalc de extremos y Ra en MJ/m²/día.
No utiliza radiación medida ni viento, y no se calibra contra EtPMon.
La [decisión científica](docs/valiantzas_v1.md) documenta por qué no se traslada
literalmente el coeficiente 0.004563 de la ecuación 4 a nuestra Ra energética.

```powershell
python -m siar_science calculate-valiantzas --dataset ./derived/observational-v1.json --output ./runs/v1-val
python -m siar_science evaluate-classical --dataset ./derived/observational-v1.json --predictions ./runs/v1-baselines/predictions.json --predictions ./runs/v1-val/predictions.json --output ./runs/v1-classical
```

La segunda operación reutiliza predicciones oficiales PM/HS verificadas y
conserva sus productores y hashes. Exige los tres métodos completos y sin
solapamientos. `run-classical-baselines` permite calcular los tres juntos sobre
entradas controladas o en una reproducción nueva. `run-baselines` y `evaluate`
conservan la semántica PM/HS y `paired` anterior.

La comparación nueva publica `own`, `PM_HS`, `PM_VAL`, `HS_VAL` y
**`PM_HS_VAL`**, esta última principal para comparar los tres en las mismas
fechas/estaciones. Cada intersección informa los modelos que la componen.
Hay desgloses por estación, año, mes, temporada, estación-año y temporada-año,
con denominadores y hashes de cohortes explícitos.

`classical-baselines.csv` conserva todas las filas y 23 columnas originales,
predicciones, motivos/flags y procedencia. UTF-8 con BOM, separador coma,
decimal punto, `date` ISO, booleanos `true/false`, ausencias vacías y
listas/objetos como JSON entrecomillado. `Fecha` original se preserva.
En Excel español, usar **Datos → Desde texto/CSV** y elegir coma; en Power
Query interpretar los decimales con configuración regional inglesa. En pandas:
`read_csv(path, encoding="utf-8-sig")`. No se redondea ni rellena con ceros.
El CSV facilita la inspección; los JSON siguen siendo los artefactos científicos.
El manifest registra hash, formato, número de filas y configuración.


## Cuarto baseline: Hargreaves con Rs medida

`HG_RS_V1` usa **0.0135×0.408×Radiacion×(Tcalc+17.8)**, con Rs medida en
MJ/m²/día y salida mm/día. Es independiente de HS_RA_V1: no usa Ra ni √TD.
No requiere humedad ni viento; sí extremos térmicos para Tcalc y Rs válida.
[Fórmula, fuentes, dominio, diferencias legacy y límites](docs/hargreaves_rs_v1.md).

```powershell
python -m siar_science calculate-hargreaves-rs --dataset ./derived/observational-v1.json --output ./runs/v1-hgrs
python -m siar_science evaluate-four --dataset ./derived/observational-v1.json --predictions ./runs/v1-classical/predictions.json --predictions ./runs/v1-hgrs/predictions.json --output ./runs/v1-four
```

La evaluación importa las predicciones anteriores y conserva productores/hashes.
`run-four-baselines` permite calcular los cuatro en una ejecución nueva.
Se publican cohortes propias, seis pares, cuatro triples y la principal
**PM_HS_VAL_HGRS**, con todos los desgloses habituales. Se conservan comandos
y semántica de dos/tres métodos. El CSV de cuatro incorpora ET0_HG_RS_V1 y
eligible/evaluated/eligibility_reasons_HGRS, sin perder filas, nulos o ceros.
Cada output se escribe en un destino nuevo; Dataset v1 no se reconstruye.

## Ejecución oficial de los tres métodos — 24/10/2026 (registro preservado)

[Comparación PM/HS/VAL, métricas y CSV](results/classical-20261024T034529Z/README.md).
VALIANTZAS-V1-AND-CLASSICAL-BASELINES-EXPORT = PASS; Dataset v1 sin cambios.
PM/HS importados de su ejecución oficial anterior; el productor original se conserva.


## Ejecución oficial de cuatro métodos — 24/10/2026

[HG_RS_V1 y comparación ampliada](results/four-classical-20261024T042250Z/README.md).
70978 predicciones HG-Rs; 70912 evaluables; cohorte
común de cuatro 70860. Dataset v1 y resultados anteriores intactos.

## Siguiente gate: preservación y metodología AHC

**PROJECT-ARTIFACT-BACKUP-V1 pendiente**: el dataset, predictions y CSV científicos masivos permanecen en el workspace Windows declarado por 05, no íntegramente en Git ni necesariamente en el servidor. [DATA_LOCATIONS](../docs/DATA_LOCATIONS.md) identifica bytes, índices y hashes. No se afirma backup completo por disponer de evidencia compacta versionada.

Secuencia posterior: revisión AHC → contrato de evaluación temporal → modelos calibrados simples → regresión/árboles/boosting → ANN. Los cuatro baselines siguen sin calibración local y Dataset v1/raw/Quality intactos. El interés es contrastar accuracy against EtPMon con input requirements / operational availability; más inputs no garantizan mejor modelo ni causalidad.

## Experimento AHC temporal versionado

[Metodología y bibliografía](docs/ahc_methodology.md),
[protocolo congelado antes del primer ajuste](experiments/ahc-v1.json),
[contrato ML, sin entrenamiento](docs/ml_experiment_contract.md) y
[contrato de tablas para visualización](docs/visualization_contract.md).
AHC_VALIDATED estima por estación un multiplicador OLS sin intercepto contra
PM_FAO56_V1 usando exclusivamente TRAIN (2004–2015). VALIDATION (2016–2020)
puede seleccionar complejidad entre ajustes ya congelados; TEST (2021–2025)
evalúa sin reajuste. TEST es retrospectivo: los baselines del histórico completo
ya fueron inspeccionados. La climatología posterior se identifica separadamente.

```powershell
python -m siar_science ahc-train --dataset ./derived/dataset.json --baseline-manifest ./runs/four/manifest.json --protocol scientific_pipeline/experiments/ahc-v1.json --output ./runs/ahc/train
python -m siar_science ahc-validate --dataset ./derived/dataset.json --baseline-manifest ./runs/four/manifest.json --protocol scientific_pipeline/experiments/ahc-v1.json --parent-manifest ./runs/ahc/train/manifest.json --output ./runs/ahc/validation
python -m siar_science ahc-test --dataset ./derived/dataset.json --baseline-manifest ./runs/four/manifest.json --protocol scientific_pipeline/experiments/ahc-v1.json --parent-manifest ./runs/ahc/validation/manifest.json --output ./runs/ahc/test
python -m siar_science ahc-closeout --dataset ./derived/dataset.json --baseline-manifest ./runs/four/manifest.json --protocol scientific_pipeline/experiments/ahc-v1.json --parent-manifest ./runs/ahc/test/manifest.json --output ./runs/ahc/closeout
```

Las cuatro etapas requieren el mismo código y entradas verificadas, destinos
nuevos y manifest padre SUCCESS. Un STOP científico preserva evidencia y
prohíbe continuar. No se reconstruye Dataset v1. Las predicciones clásicas se
importan con sus hashes y productores; las tablas derivadas registran ambas
identidades SHA de fuentes, pipeline, protocolo, coeficientes y cohortes.

## Resultado oficial AHC — 24/10/2026

[AHC-METHODOLOGY-AND-TEMPORAL-VALIDATION = PASS](results/ahc-20261024T172400Z/README.md).
TRAIN 2004–2015, VALIDATION 2016–2020 y TEST 2021–2025, sin refit.
En TEST común contra PM (n=15.958), RMSE anual ajustado HS 0.638838,
HG-Rs 0.414241 y VAL 0.523955 mm/día. HG-Rs estacional seleccionado en
VALIDATION: 0.401392. Estabilidad por bloques PASS; climatología separada,
IB101 descriptivo no desplegable e IB09 excluida. Se retienen empeoramientos
estación-año y condición retrospectiva de TEST. ML y web siguen sin ejecución.
