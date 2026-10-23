# SIAR Science — ET₀ baselines v1

Capa reproducible de observaciones SIAR Baleares: validación científica por método, Penman–Monteith diario FAO-56, Hargreaves–Samani fijo y evaluación frente a EtPMon. Conserva valores originales y genera resultados derivados separados. No incorpora forecast, ML, AHC ni limpieza global ±3σ.

La forense queda **RESOLVED_FOR_V1** por decisión de Dirección 00 del 2026-10-19. Radiacion se interpreta directamente en MJ m⁻² día⁻¹ y VelViento en m/s, con convención SIAR de viento a 2 m. `UNITS_PROVENANCE_PARTIAL` y `station_period_provenance_not_exhaustively_verified` son limitaciones registradas, no bloqueos. Una evidencia positiva de altura distinta de 2 m requiere resolver el cálculo PM afectado.

**ET0-BASELINES-PREP = READY_FOR_DATASET_V1.** Implementación integrada, 48 pruebas matemáticas/sintéticas locales PASS y CI científica PASS con paquete instalado en Python 3.11, 3.12 y 3.13. Las evidencias de validación constan en el [estado de misión](docs/mission_status.md). No se ha generado el Dataset Observational v1 definitivo: su construcción y ejecución oficial esperan Quality POST y handoff autorizado de 00.

## Instalación y ayuda

Python 3.11 o posterior. Desde la raíz del repositorio, en un entorno Python de trabajo:

```powershell
python -m pip install -e ./scientific_pipeline
python -m siar_science --help
```

No se necesitan credenciales SIAR ni acceso a la API para ejecutar la capa sobre entradas locales autorizadas. La interfaz está implementada y dispone de ayuda por comando con `--help`.

## Flujo previsto

Los siguientes son ejemplos de uso futuro; las rutas son ilustrativas y no indican que los artefactos finales ya existan. No ejecutar la construcción definitiva hasta recibir el handoff de cierre Quality. Los outputs deben ser rutas nuevas.

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

Cada comando de ejecución crea un directorio nuevo con:

| Archivo | Contenido |
|---|---|
| `predictions.json` | Resultados por estación/fecha/método, elegibilidad, motivos y flags; `null` cuando no se calcula |
| `metrics.json` | Cohortes propias y pareadas, tamaños, métricas y agrupaciones; valores no definidos como `null` |
| `manifest.json` | Identidades/hashes de entrada y outputs, versión/SHA del pipeline, configuración, políticas, limitaciones y estado de ejecución |

Validación conserva las filas y sus metadatos de elegibilidad, con ET₀ sin calcular y la colección de métricas vacía. No se permite sobrescribir una ejecución anterior. Se preservan EtPMon y TempMedia SIAR, flags de origen y referencias POST. Una ejecución sintética verifica software, no crea Dataset v1 ni certifica observaciones reales.

## Decisiones científicas

Ambos baselines usan Tcalc=(Tmax+Tmin)/2, conservando TempMedia original. PM fija G=0, γ=0.000665P y albedo=0.23; calcula presiones de vapor con extremos térmicos y de humedad. La nubosidad de Rnl limita Rs/Rso por arriba a 1 y **no aplica floor inferior por defecto**. Los ratios extremos, Rnl negativo y ET₀ negativa se marcan; no se recorta el resultado a cero.

HS usa el coeficiente fijo **0.0023·0.408**, Ra calculada, temperatura y rango térmico. No usa Rs medida ni coeficientes ajustados. La elegibilidad depende de sus propias entradas: humedad o radiación medidas ausentes pueden impedir PM y permitir HS; EtPMon ausente impide evaluación y permite predicción.

Error=predicción−EtPMon. Se publican n, MAE, RMSE, MBE/bias y R² predictivo (1−SSE/SST), conservando los ceros de referencia. La referencia evaluable es finita, no negativa y procede de una observación aceptada dentro de periodo oficial; los valores negativos se preservan con flag y se excluyen de métricas. Se distinguen cohortes propias de cada modelo y la intersección pareada. EtPMon es una estimación SIAR de comparación, no verdad independiente de terreno. No se presentan las métricas históricas del legado como resultados de esta nueva implementación.

PM/HS no aprenden parámetros. Cualquier calibración o ML futuro requiere separación cronológica global TRAIN<VALIDATION<TEST; filtros aprendidos, escalas y AHC se ajustan solo en TRAIN. No se reutiliza la calibración histórica dentro de muestra ni se divide aleatoriamente.

## Referencias de la entrega

- [Primera ejecución oficial PM/HS: resultados, cohortes, hashes y limitaciones](results/pm-vpd-20261023T194553Z/README.md).
- [Supuestos y diferencias con el legado](docs/assumptions.md).
- [Contrato de dataset, handoff y ejecución](docs/contracts.md).
- [Estado de misión y condiciones pendientes](docs/mission_status.md).
- [Auditoría del legado](docs/legacy_audit.md).
- [Evidencia offline de unidades](docs/units_offline_evidence.md).
- [Resultados numéricos forenses preservados](docs/units_offline_forensic.md), cuyo diagnóstico preliminar se interpreta con la decisión posterior de 00.

Pruebas locales: `python -m unittest discover -s scientific_pipeline/tests -v`. La suite matemática y sintética verifica fórmulas publicadas, QC/eligibilidad, métricas, handoff/POST/catálogo, hashes, protección de outputs, procedencia y CLI. La CI científica ejecuta el paquete instalado y la CLI en Python 3.11, 3.12 y 3.13. El cierre y las evidencias verificadas se registran en el estado de misión.
