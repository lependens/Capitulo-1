# Recorrido científico y cronológico

Esta guía ordena las fuentes; no sustituye los protocolos ni convierte estados históricos en resultados nuevos. Para el relato técnico completo, ecuaciones, bibliografía y limitaciones, véase la [memoria científica V2](../scientific_pipeline/docs/ET0_V2_MEMORIA_CIENTIFICA.md).

## De los datos a los experimentos

1. **Origen y observaciones SIAR.** El trabajo parte del TFG y de estaciones de Baleares: [objetivos históricos](../docs_1.0_Objetivos%20alcanzables.md), [obtención](../docs_1.1_Obtenci%C3%B3n%20de%20datos.md) y [datos de Baleares](../datos_siar_baleares/). Los CSV evolutivos [V3](../datos_siar_v3/) y Sync tienen otro ciclo de actualización; no son automáticamente el dataset congelado.
2. **Quality POST.** El [informe del cierre](../data_quality/siar/post/SIAR-QA-POST-20261020T193140Z/final_consolidated_report.md) y la [corrección del ledger](../data_quality/siar/ledger-correction.md) documentan completitud, ausencias y límites; el POST anterior se conserva. COMPLETE no significa ausencia de datos faltantes. Véanse la [adjudicación IB09](../data_quality/siar/ib09-v2-adjudication.md) y el [handoff DEVELOPMENT](../data_quality/siar/ib09-handoff-v2-development/decision.md).
3. **Dataset Observational v1.** Se construye a partir de la evidencia y reglas documentadas; su identidad y ubicación se describen en [datos y ubicaciones](DATA_LOCATIONS.md) y [pipeline](../scientific_pipeline/README.md). No confundirlo con CSV que siguen cambiando.
4. **Referencia y fórmulas.** PM FAO-56 calculada es el target de las comparaciones señaladas; HS-Ra, HG-Rs y Valiantzas usan diferentes entradas. Consultar [memoria y ecuaciones](../scientific_pipeline/docs/ET0_V2_MEMORIA_CIENTIFICA.md), [HG-Rs](../scientific_pipeline/docs/hargreaves_rs_v1.md), [Valiantzas](../scientific_pipeline/docs/valiantzas_v1.md) y [supuestos](../scientific_pipeline/docs/assumptions.md). Las comparaciones históricas frente a EtPMon deben conservar su referencia explícita.
5. **Calibración AHC.** Ajusta un multiplicador de una estimación simplificada frente a PM; periodos y cohortes importan. Ver [metodología](../scientific_pipeline/docs/ahc_methodology.md). El histórico completo in-sample no es TEST independiente.
6. **ML y ANN V1.** La [guía ML clásico](../scientific_pipeline/docs/ml_v1_classical_methodology.md), el [incidente de acceso TEST](../scientific_pipeline/docs/ml-v1-test-recovery-access-incident.md) y la [guía ANN moderna](../scientific_pipeline/docs/ann_v1_modern_explained.md) conservan diseño, recuperación y límites. La recuperación retrospectiva no demuestra una única apertura de TEST.
7. **V2: DEVELOPMENT y freeze.** La [metodología y reproducción](../scientific_pipeline/docs/et0_v2_metodologia_y_reproduccion.md), [decisiones](../scientific_pipeline/docs/et0_v2_decisions.md) y [ejecución](../scientific_pipeline/docs/et0_v2_execution.md) describen selección, celdas, semillas y congelación previa a evaluación. Los [metadatos experimentales](../scientific_pipeline/docs/et0_v2_experimental_metadata.md) describen DEVELOPMENT, sin crear TEST para candidatos descartados.
8. **Retrospectiva y revisión.** Los [resultados](../scientific_pipeline/docs/et0_v2_retrospective_results_2026.md), [certificación técnica](../scientific_pipeline/docs/et0_v2_certificacion_evidencia.md) y [contrato de inferencia](../scientific_pipeline/docs/et0_v2_contrato_inferencia.md) precisan cobertura, cohortes y límites. V2 sigue NOT_PUBLISHED; la evaluación 2020–2025 no es prospectiva ni medición instrumental independiente.
9. **Preservación y presentación.** Consultar [recuperación](backup/README.md), [ubicaciones](DATA_LOCATIONS.md), [web V2 estable](../web_app/docs/et0-v2-stable.md) y [V0.3](../web_app/docs/release-v03.md). Cada recibo o restore acredita solo el alcance y fecha documentados.

La [auditoría científica independiente de ET₀ V2, TFG y TFM](../scientific_pipeline/docs/et0_v2_auditoria_cientifica_tfg_tfm.md), fechada el 25/11/2026 e incorporada en la PR #87, compara los antecedentes con el experimento V2. Encuentra concordancia descriptiva del orden y escala de errores, con periodos, cobertura y referencias diferentes. Recoge posibles errores en código histórico impreso, cuya incidencia en los resultados originales requiere recuperar las versiones ejecutadas y sus datos. La comparación diaria con una referencia común y la revisión de la altura del viento siguen pendientes. V2 continúa NOT_PUBLISHED.

## Líneas operativas y futuras

[Sync](../siar-sync/README.md) mantiene observaciones; [SIAR Forecast](../forecast_collector/README.md) y [AEMET Forecast](../aemet_forecast_collector/README.md) acumulan previsiones. Estas líneas no son el dataset congelado ni una evaluación ET₀ por horizonte. La inferencia con previsiones y la validación prospectiva figuran en la [hoja de ruta](ROADMAP.md), sin declararlas realizadas.

## Documentos históricos de la raíz

Las rutas existentes se conservan para compatibilidad. Sus estados y planes fechados pueden estar superados:

- [1.0 Objetivos](../docs_1.0_Objetivos%20alcanzables.md) y [1.1 Obtención](../docs_1.1_Obtenci%C3%B3n%20de%20datos.md).
- [1.2 Tratamiento](../docs_1.2_Tratamiento%20de%20datos%20y%20c%C3%A1lculo%20de%20estimaciones.md) y [1.2.6 ML clásico](../docs_1.2.6_Modelos%20ML%20cl%C3%A1sicos%20para%20estimaci%C3%B3n%20de%20ET0.md).
- [1.3 Errores](../docs_1.3_An%C3%A1lisis%20errores.md), [1.4 Redes](../docs_1.4_Modelos%20redes%20neuronales.md) y [2.1 ANN TensorFlow](../docs_2.1_Estimaci%C3%B3n%20Et0%20ANN%20%28tensorflow%29.md).

Consultar [hitos](MILESTONES.md), [estado](PROJECT_STATUS.md) y [auditoría del 26/11](AUDITORIA_DOCUMENTAL_2026-11-26.md) para fuentes posteriores.
