# ET0-BASELINES-PREP — estado de integración

Fecha: 2026-10-19 (Europe/Madrid). Decisión de Dirección 00: cerrar forense y avanzar preparación integrada.

## Estado vigente

- Forense: **RESOLVED_FOR_V1**.
- Interpretación aceptada: Radiacion MJ m⁻² día⁻¹; VelViento m/s; lectura directa, sin ×0.0864 ni /3.6.
- Viento: convención SIAR de 2 m aceptada para v1. La falta de historial instrumental exhaustivo no es STOP.
- Limitaciones no bloqueantes: **UNITS_PROVENANCE_PARTIAL** y **station_period_provenance_not_exhaustively_verified**, conservadas en trazabilidad/manifiestos.
- Evidencia positiva de altura distinta de 2 m: detener PM en el ámbito afectado hasta resolución explícita; no bloquea automáticamente HS.
- Objetivo: **READY_FOR_DATASET_V1**, condicionado a completar y verificar integración, pruebas/CI y revisión. No se certifica aquí PASS final ni merge.

Esta decisión supera los diagnósticos preliminares BLOCKED_UNITS y WIND_HEIGHT_UNVERIFIED de la fase anterior. No reescribe las cifras forenses ni transforma una convención aceptada en verificación instrumental por estación/periodo. No se continuarán búsquedas históricas para avanzar esta misión.

## Alcance preservado

La investigación partió de main `6f1dd95c415730ec29843055d3a32a7ac5a636ad` y documentación 1.2, seguida de código, outputs legacy y documentación 1.3. La auditoría y el contraste offline quedan como evidencia histórica. El informe forense generado conserva sus valores y diagnóstico fechado; una nota posterior de decisión 00 define la interpretación operativa actual.

No se ha ejecutado el Dataset Observational v1 definitivo ni publicado resultados finales de baselines sobre él. No se modifica adquisición, Quality, raw, CSV canónicos, legacy o el esquema canónico de 23 columnas. Las pruebas sintéticas no sustituyen Quality POST.

## Preparación integrada requerida

La entrega incorpora el contrato de CLI `python -m siar_science` con validación, construcción desde handoff, PM, HS y evaluación; salidas nuevas con predicciones, métricas y manifiesto. El cierre técnico exige comprobar:

1. Funciones FAO-56/HS con ejemplos independientes, dominios y políticas numéricas registradas.
2. Adaptador del handoff `SIAR-QUALITY-CLOSEOUT`, esquemas POST oficiales y hashes de fuentes e informes, sin fabricar cierre autorizado.
3. QC por método, preservación de observaciones y flags, referencia SIAR separada de predicción.
4. Ejecución sintética de extremo a extremo y pruebas de rechazo de contradicciones, duplicados y sobrescritura.
5. Trazabilidad de dataset/configuración/código/cohortes/outputs en manifiesto y CI científica verificable.

El [README](../README.md), [contrato](contracts.md) y [supuestos](assumptions.md) documentan el comportamiento previsto. No se declara aprobado ningún check final antes de obtener su resultado real. La identidad del commit/merge y los checks finales pertenecen al cierre técnico consolidado; no se inventan aquí.

## Decisiones científicas congeladas para v1

PM diario FAO-56 y HS sin calibración usan media de extremos. PM fija G=0, γ estándar, albedo=0.23 y VPD de extremos; Rnl no tiene floor inferior por defecto. Conservar negativos con flags, sin sustituciones silenciosas ni ±3σ global. HS usa 0.0023·0.408 fijo y Ra. EtPMon sigue como referencia calculada SIAR, no verdad independiente.

Métricas: n, MAE, RMSE, MBE/bias, R² predictivo; cohortes propias y pareadas separadas, referencias cero conservadas. Futura calibración/ML exige cortes temporales globales y aprendizaje solo en TRAIN; no forma parte de esta misión.

## Gate de ejecución definitiva

Preparar y verificar el software no autoriza generar el dataset final ahora. La construcción definitiva espera el handoff autorizado de 00 después de Quality closeout: `gate=SIAR-QUALITY-CLOSEOUT`, informes POST originales, catálogo/metadatos, hashes, unidades y fuentes identificadas. El mismo gate sigue vigente aunque la forense esté resuelta.

Tras comprobar integración y CI podrá registrarse READY_FOR_DATASET_V1 para la capa. La disponibilidad de Quality POST y Dataset v1 es un estado separado; la misión no los declara completados ni ejecuta los baselines definitivos para anticiparlos.
