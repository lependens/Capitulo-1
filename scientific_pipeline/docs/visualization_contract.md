> **Estado posterior — 31/10/2026:** WEB-V0.2-AHC ya está desplegada (PR #46/#47) y el runtime ML rolling está implementado (PR #48); rolling validation y Linux smoke PASS. La campaña científica oficial permanece pendiente de despliegue/ejecución; ANN NOT STARTED. Las referencias inferiores a «web no construida», ML futuro o merge pendiente conservan el estado previo de su entrega. No cambian la semántica contractual ni resultados congelados. Para el contrato rolling vigente, véase [refinamiento temporal](ml_validation_contract_refinement.md); estado rápido en [PROJECT_STATUS](../../docs/PROJECT_STATUS.md).

# Contrato de datos para visualización científica v1

**Estado documental vigente — 25/10/2026: FROZEN / AVAILABLE TO 06.** AHC temporal ya está PASS. Este contrato describe consumo/experimentos futuros: no ML/ANN entrenado ni web construida. El consumidor utiliza las tablas/schemas del productor y no recalcula ciencia ni inventa agregados no publicados. [Estado maestro](../../docs/PROJECT_STATUS.md).

El productor AHC genera una vista derivada para un futuro consumidor Web; esta misión **no construye la web**. Este contrato describe archivos del productor, sin afirmar que una ejecución concreta haya pasado: comprobar status, manifest y hashes del lote. Dataset, raw, Quality y baselines permanecen congelados. Ver [metodología AHC](ahc_methodology.md) y [contrato ML](ml_experiment_contract.md).

## Paquete, esquema y formato

El paquete incluye `view-manifest.json` independiente y `view_schemas.json` con los encabezados y tipos de las tablas realmente escritas. El manifest de ejecución AHC conserva además el linaje completo. Validar versión, estado, dataset, protocolo y hashes antes de consumir tablas. Los campos exactos y su orden se obtienen del esquema: no inferir columnas opcionales ni sustituir mapas JSON por columnas inventadas.

CSV UTF-8 BOM, coma, decimal punto, fecha ISO y booleanos `true/false`. Null es celda vacía; cero es `0`. No redondear valores almacenados. Dicts/listas se serializan como JSON compacto en celdas CSV citadas. Parquet es una posible extensión futura, no una salida obligatoria ni dependencia de v1.

Las tablas transportan la identidad común declarada por su esquema: dataset_id, dataset_sha256, audit_source_repo_sha, source_repo_sha, protocol_sha256 y run_id. POST/catálogo, fuentes, pipeline, parámetros, timestamps y productores upstream quedan enlazados en manifests. El importador AHC no se atribuye el código de los baselines originales. Los hashes identifican bytes escritos; el hash del propio manifest se publica fuera de ese documento para evitar autorreferencia.

## daily_predictions.csv

Una fila por observación preservada, clave `dataset_id, station_code, date`. Dataset v1 conserva 75.496 filas, incluida IB09/2005-10-23 con OUTSIDE_OFFICIAL_PERIOD. El ledger de huecos no crea meteorología ficticia.

Además de identidad común, las columnas son:

```text
date, station_code, split, EtPMon
PM_FAO56_V1, HS_RA_V1, HG_RS_V1, VALIANTZAS_V1
HS_RA_AHC_VALIDATED, HG_RS_AHC_VALIDATED, VALIANTZAS_AHC_VALIDATED
eligibility, evaluated_PM, evaluated_SIAR, coefficient_ids
observation_status, quality_status, quality_flags, scientific_flags
source_data_sha, provenance
```

Las columnas de modelo proceden del conjunto de predicciones del lote. Una variante seleccionada en VALIDATION tiene columna adicional con sufijo `_SELECTED`; no reemplaza el ajuste anual primario ni se incorporan candidatos descartados como selecciones TEST.

Los siguientes campos son **celdas JSON**, no familias de columnas planas `eligible_*` o `evaluated_*`:

- `eligibility`: información de elegibilidad/motivos conservada por el productor.
- `evaluated_PM`: mapa modelo → pareja evaluable con PM_FAO56_V1.
- `evaluated_SIAR`: mapa modelo → pareja evaluable con EtPMon.
- `coefficient_ids`: referencias a coeficientes de ajustes aplicados.
- `quality_flags`, `scientific_flags` y `provenance`: estructuras conservadas; provenance identifica archivo/fila fuente.

Un hash de fuente no es un hash de registro. Predicción ausente es null, nunca cero ni sustitución por el baseline original. Elegibilidad de inputs, disponibilidad de coeficiente y pareja evaluable son conceptos distintos. TRAIN ajustado es dentro de muestra. TEST es **retrospective_locked** porque los baselines históricos ya se examinaron; no constituye confirmación prospectiva ni un periodo nunca visto.

No se duplican las 23 columnas meteorológicas. Añadir variables para otra vista requiere versión de esquema o enlace al dataset por identidad/claves. AHC_CLIMATOLOGY nunca ocupa una columna AHC_VALIDATED; sus coeficientes se entregan separadamente tras el cierre.

## Métricas precalculadas

`metrics.csv` contiene todos los grupos. Los archivos especializados seleccionan registros de `experimental_metrics.py`:

| Archivo | Valores de group |
|---|---|
| station_metrics.csv | station |
| monthly_metrics.csv | year_month, station_year_month |
| seasonal_metrics.csv | season_year, station_season_year |
| yearly_metrics.csv | year, station_year |
| climatological_month_metrics.csv | month |
| climatological_season_metrics.csv | season |

Los campos científicos, además de identidad común, son:

```text
split, reference_id, group, key, station_code
cohort_id, cohort_sha256, method_id, model_instance_id
start_date, end_date
n, mae, rmse, bias, r2, r2_status
```

No renombrar `group/key` a period_key ni deducir identidad a partir del nombre del archivo. start_date/end_date describen las fechas de las parejas seleccionadas; un grupo vacío puede carecer de fechas. reference_id distingue PM y SIAR. `own`, `paired:<base>`, `common_7` y, cuando existen, `selected:<base>` y `common_selected` son cohortes diferentes. El hash representa claves seleccionadas.

`month` agrupa un mes climatológico de la partición; `year_month` es un año-mes concreto. `season` y `season_year` también se distinguen. DJF asigna diciembre al año siguiente; split sigue siendo parte de la agrupación, por lo que diciembre 2020 y enero-febrero 2021 no se fusionan cruzando VALIDATION/TEST. Bordes parciales no se completan artificialmente.

Métricas: n, MAE, RMSE, bias=predicción−referencia y R²=1−SSE/SST. Referencias cero permanecen. r2 null lleva r2_status y nunca se representa como cero. El global agrupa parejas, sin promediar RMSE/R² de estaciones. No sumar n de cohortes solapadas ni presentar cohortes propias distintas como pareadas.

## Metadatos de modelos y coeficientes

`model_metadata.csv` identifica method_id/model_instance_id y conserva base_method_id, version, reference_id, input_parameters (JSON), adjustment_kind, calibration_split, selection (JSON) e interpretation, más identidad común y los campos reales de view_schemas.json. Los baselines pueden tener campos de ajuste null; los ajustes validados usan TRAIN. `_SELECTED` conserva la selección congelada. La interpretación sigue siendo emulación de referencia calculada, no ET medida.

`ahc_coefficients.csv` utiliza estos nombres:

```text
coefficient_id, station_code, base_method_id, kind, reference_id
granularity, group_key, coefficient, status, deployable, fit_split
calibration_start, calibration_end, fit_key_sha256, support, estimate
```

`kind` distingue AHC_VALIDATED/AHC_CLIMATOLOGY. `support` es JSON de soporte de muestras/años/robustez. `estimate` es JSON y contiene `estimator`, coefficient y recuentos/diagnósticos; no se promete un campo plano estimator_id. OLS0 es adimensional: Σ(x·PM)/Σ(x²). group_key identifica all, temporada o mes según granularidad. Un coeficiente ausente conserva status/motivo, sin fallback regional.

`ahc_validated_coefficients.csv`, donde se emite, usa el mismo esquema para coeficientes TRAIN congelados. `ahc_climatology_coefficients.csv` aparece en cierre posterior a TEST: no sobrescribe el validado ni demuestra rendimiento histórico fuera de muestra. Series cortas conservan soporte y deployable=false cuando no alcanzan robustez.

## station_metadata.csv

Campos de estación, además de identidad común:

```text
station_code, station_name, latitude_deg, altitude_m
official_start, official_end
source_row_count, first_source_date, last_source_date
source_quality_statuses, available_splits
station_metadata, catalog_metadata, catalog_sha256
```

Los campos estructurados son JSON. Periodos oficiales no se deducen de fechas extremas de filas. Nombre/periodos provienen de evidencia autorizada; desconocido es null. Longitud del catálogo permanece en formato nativo dentro de catalog_metadata: **no se inventa longitude_deg decimal**. Convención SIAR de viento a 2 m y procedencia no exhaustiva permanecen en station_metadata/linaje, sin declararlas certificadas estación-periodo.

## Cobertura, complejidad y asignación temporal

`coverage_metrics.csv` presenta conteos de disponibilidad, predicción y evaluación por estación, partición y modelo. Tener inputs no equivale a disponer de coeficiente robusto. Motivos solapados no forman una partición exclusiva de filas.

`input_complexity.csv` declara requisitos de inputs de PM y los empíricos, y el ajuste TRAIN adicional cuando corresponde. Permite presentar precisión frente a complejidad/disponibilidad, sin atribuir causalidad al comparar cohortes propias distintas.

`split_assignments.csv` se emite en TRAIN y fija claves estación-fecha y partición global; su hash lógico `split_assignment_sha256` vincula análisis y tablas, mientras el hash de bytes está en el manifest TRAIN enlazado por la cadena de padres. Sus columnas son station_code, date, split, dataset_id y dataset_sha256. No se duplica en el lote TEST ni se incluye en su view_schemas.json. Las fechas exteriores conservan su estado y siguen siendo inelegibles, aunque su fecha pertenezca a una partición experimental. Los campos/tipos de las tablas del lote TEST están en view_schemas.json; no hay columnas adicionales implícitas.

## Responsabilidad del consumidor y extensiones

La interfaz puede ordenar, filtrar y dibujar valores publicados. No recalcula ET₀, AHC, elegibilidad, cohortes o métricas; no reajusta parámetros ni decide exclusiones. Debe mostrar referencia, split, cohorte, n, carácter retrospectivo y limitaciones. Para un filtro sin agregado publicado, mostrar no disponible o solicitar otro artefacto científico, sin promediar métricas para simularlo.

Una tabla de huecos, Parquet, features meteorológicas adicionales o columnas de climatología diaria son extensiones futuras explícitas, no obligaciones cumplidas por los CSV actuales. view_schemas.json y el manifest del lote son la autoridad sobre archivos efectivamente generados.
