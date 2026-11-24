# ET₀ V2: handoff de lectura — catálogo técnico VERIFIED

El catálogo técnico definitivo está verificado y restaurado independientemente. La publicación oficial sigue NOT_PUBLISHED y requiere aprobación de Dirección. No recalcular modelos desde la web.

- Original íntegro: `/srv/siar-project/scientific-artifacts/et0-v2/original/retrospective-20261116/et0-v2-retrospective-370251881.zip`, SHA256 `2fddd95c2374a95b4bd1885a6d13f4455ed89d45b8e0d9e854e4d846d40cd420`.
- Datos derivados y catálogo anterior DRAFT: `/srv/siar-project/scientific-artifacts/et0-v2/derived/v1/`; `catalog.json` SHA256 `f497074a39d25a38d637329a87a652c470875502a2c9f796cf2eca6706148d81`.
- Predicciones: `parquet/station_id=IB01/year=2020/predictions.parquet` y equivalentes por estación/año; `csv-partitions/IB01-2020.csv` y equivalentes.
- Métricas: `retrospective-metrics.csv`. AHC histórico completo: `ahc-full-history-in-sample.csv`.

Predicciones: `station_id,date,method_id,method_family,resolution,prediction,reference_value,reference_id,unit,cohort_id,eligibility_status,quality_status,observation_status,formula_status,formula_reasons,pm_reasons,source_run_id,freeze_sha256,dataset_sha256,schema_version`. `eligibility_status` indica presencia de salida, no sustituye el QC científico. `common_core` identifica la intersección fórmula/AHC/ML principal/ANN A/ANN E; las variantes ML requieren su propio pareado.

Métricas: `family,calendar,group,key,method,calendar_days,pm_days,predicted_days,pm_pairs_own,coverage_calendar,coverage_pm_own,common_keys_sha256,n,mae,rmse,bias,r2`. La cobertura es propia y el error usa la cohorte común indicada; no son el mismo denominador. No inferir intervalos temporales por grupo si no están disponibles en la tabla.

Estaciones retrospectivas: IB01, IB02, IB03, IB04, IB05, IB06, IB08, IB10 e IB11, entre 2020-01-01 y 2025-12-31 según disponibilidad. IB07, IB09 e IB101 son históricas y no tienen observaciones retrospectivas. Familias: HS_RA_V1, HG_RS_V1, VALIANTZAS_V1; resoluciones: annual, seasonal, monthly. PM_FAO56_V1 es referencia calculada; AHC de histórico completo es in-sample.

Ejemplo SQL sobre una vista registrada `predictions`: `SELECT date, method_id, prediction, reference_value FROM predictions WHERE station_id='IB01' AND method_family='HG_RS_V1' AND resolution='annual' AND date BETWEEN '2020-01-01' AND '2020-12-31';` La aplicación debe leer únicamente una versión del catálogo explícitamente aprobada y verificar su identidad.


**Advertencia para interfaz:** Existen predicciones negativas de ANN y de regresiones lineales. Deben mostrarse con una indicación de que son salidas no restringidas y no representan necesariamente ET₀ física; conservar los valores y métricas sin truncarlos. Véanse los recuentos y ejemplos en [evidencia de certificación](et0_v2_certificacion_evidencia.md). No presentar `common_core` como cohorte idéntica de todas las variantes ML sin comprobar su intersección específica.

## Catálogo técnico verificado

Se ha preparado un catálogo separado en `/srv/siar-project/scientific-artifacts/et0-v2-verified-catalog-20261124/catalog.json`, SHA256 `fef85f0a6ae981792d867a9de2b10b9f7787bc680461820d2284cd20453df4ab`. El inventario del paquete fue firmado por el propietario; el servidor comprobó su firma y los hashes. La copia remota `backup_20261124T160201Z_5a69c7b9` consta como verificada y el catálogo se restauró independientemente con hash idéntico. `server_root` señala los mismos derivados en `et0-v2/derived/v1/`; el catálogo DRAFT original y los datos no se modificaron. El catálogo verificado tiene `publication_status=NOT_PUBLISHED`: su existencia no sustituye la aprobación de Dirección para publicación.
